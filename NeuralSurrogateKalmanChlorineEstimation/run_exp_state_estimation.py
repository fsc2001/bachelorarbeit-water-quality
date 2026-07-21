"""
This module implements the experiments for integrating the neural surrogate model into the Extended Kalman Filter.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from epyt_flow.simulation import ScadaData
from epyt_flow.utils import plot_timeseries_data, plot_timeseries_prediction
from epyt_control.evaluation.metrics import mape
from epyt_control.signal_processing.state_forecasting import DnnStateTransitionModel
from epyt_control.signal_processing.state_estimation import TimeVaryingExtendedKalmanFilter


def create_random_sensor_placement(n_node_quality_sensors: int, n_link_sensors: int, n_nodes: int,
                                   n_links: int, state_dim: int) -> np.ndarray:
    nodes_idx = random.sample(range(n_nodes), k=n_node_quality_sensors)
    links_idx = random.sample(range(n_links), k=n_link_sensors)

    nodes_idx.sort()    # We want to create a binary mask for the input!
    links_idx.sort()

    M = np.zeros((n_node_quality_sensors + 2 * n_link_sensors, state_dim))
    flows_idx = []
    i = 0
    for idx in nodes_idx:
        M[i, idx] = 1
        i += 1
    for idx in links_idx:
        M[i, n_nodes + idx] = 1
        i += 1
    for idx in links_idx:
        j = n_nodes + n_links + idx
        flows_idx.append(j)
        M[i, j] = 1
        i += 1

    return M, flows_idx


class MyDnnStateTransitionModel(DnnStateTransitionModel):
    def __init__(self, n_actuators: int, **kwds):
        self.n_missing_flows = None
        self._n_actuators = n_actuators

        super().__init__(**kwds)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model.training is False:
            stop = x.size(dim=1) - self._n_actuators    # Ignore control signals!
            flows = x[:, self.n_missing_flows:stop]
            state_pred = self._model(x)
            return torch.cat((state_pred, flows), dim=1)
        else:
            return self._model(x)


def get_mlp_state_transition_model(net_desc: str):
    if net_desc == "Net1":
        return MyDnnStateTransitionModel(hidden_layers_size=[512, 512, 128],
                                         activation="relu", last_layer_activation=None,
                                         max_iter=2000, normalization_layer=False,
                                         normalize_input_output=True, n_actuators=1,
                                         dropout=.01, batch_size=1024)
    elif net_desc == "Hanoi":
        return MyDnnStateTransitionModel(hidden_layers_size=[1028, 512, 128],
                                         activation="relu", last_layer_activation=None,
                                         max_iter=2000, normalization_layer=False, n_actuators=1,
                                         normalize_input_output=True, dropout=.05,
                                         batch_size=1024)
    elif net_desc == "CY-DBP":
        return MyDnnStateTransitionModel(hidden_layers_size=[1028, 512, 128],
                                         activation="relu", last_layer_activation=None,
                                         max_iter=2000, normalization_layer=False, n_actuators=1,   # Change back for old CY-DBP
                                         normalize_input_output=True, dropout=.05,
                                         batch_size=1024)
    else:
        raise ValueError(f"Unknown network '{net_desc}'")


def get_state_transition_model(net_desc: str, file_in: str):
    model = get_mlp_state_transition_model(net_desc)
    model.load_from_file(file_in)

    return model


def run_state_estimation(net_desc: str, scada_file_in: str, control_actions_file_in: str,
                         state_transition_model_file_in: str, n_node_quality_sensors: int,
                         n_link_sensors: int) -> list[float]:
    # Load data
    scada_data = ScadaData.load_from_file(scada_file_in)
    control_actions = np.load(control_actions_file_in)["control_actions"]

    X_flows = scada_data.get_data_flows()
    X_nodes_quality = scada_data.get_data_nodes_quality()
    X_links_quality = scada_data.get_data_links_quality()
    n_time_steps = X_flows.shape[0]

    next_flow = X_flows[1:, :]
    cur_node_quality = X_nodes_quality[:n_time_steps-1, :]
    cur_link_quality = X_links_quality[:n_time_steps-1, :]

    next_state = np.concatenate((X_nodes_quality[1:, :], X_links_quality[1:, :]), axis=1)

    X_cur_state = np.concatenate((cur_node_quality, cur_link_quality, next_flow), axis=1)
    X_control = control_actions[:n_time_steps-1, :]

    state_dim = X_cur_state.shape[1]
    n_cl_items = next_state.shape[1]
    n_nodes = cur_node_quality.shape[1]
    n_links = cur_link_quality.shape[1]

    state_transition_model = get_state_transition_model(net_desc, state_transition_model_file_in)
    state_transition_model.n_missing_flows = next_state.shape[1]

    # Integrate scaling
    state_transition_model._normalize_input_output = False
    X_cur_state_ = np.concatenate((X_cur_state, X_control), axis=1)
    X_cur_state = state_transition_model._scaler.transform(X_cur_state_)[:, :X_cur_state.shape[1]]

    # Create a sensor placement
    M, flows_idx = create_random_sensor_placement(n_node_quality_sensors, n_link_sensors,
                                                  n_nodes, n_links, state_dim)
    obs_dim = M.shape[0]
    #obs_dim = state_dim
    #M = np.eye(state_dim)

    # Create Kalman filter
    measurement_func = lambda x: np.dot(M, x.flatten())
    measurement_func_grad = lambda _: M

    def get_measurement_func(t: int):
        return measurement_func

    def get_measurement_func_grad(t: int):
        return measurement_func_grad

    def get_control_signal(t: int) -> np.ndarray:
        return X_control[t+1, :].reshape(1, -1)

    def get_state_transition_func(t: int):
        x_control = get_control_signal(t)
        return lambda x: state_transition_model.predict(x.reshape(1, -1), x_control).flatten()

    def get_state_transition_func_grad(t: int):
        x_control = get_control_signal(t)

        def get_jac(x_cur_state):
            jac = state_transition_model.compute_jacobian(x_cur_state.reshape(1, -1), x_control)
            jac = jac.reshape(jac.shape[1], jac.shape[3])  # Eliminate batch dimensions!
            jac = jac[:, :state_dim]   # Remove control input

            return jac

        return get_jac


    my_filter = TimeVaryingExtendedKalmanFilter(state_dim=state_dim, obs_dim=obs_dim,
                                                init_state=X_cur_state[0, :],
                                                get_state_transition_func=get_state_transition_func,
                                                get_state_transition_func_grad=get_state_transition_func_grad,
                                                get_measurement_func=get_measurement_func,
                                                get_measurement_func_grad=get_measurement_func_grad)

    # Apply and evaluate filter
    avg_cl_scores = []
    avg_flow_scores = []

    cl_pred = []
    cl_true = []
    flows_pred = []
    flows_true = []
    cl_pred_std = []

    for i in range(1, X_cur_state.shape[0]):    # NOTE: We start with the second state as an observation!
        # Inject current flow readings
        cur_state = X_cur_state[i, :]   # Inject observed flows
        for idx in flows_idx:
            my_filter._x[idx] = cur_state[idx]

        # Apply measurement functions
        x_observation = measurement_func(X_cur_state[i, :])

        # Predict current state based on observations (and undo scaling in the output)
        cur_state_pred, cov_state_pred = my_filter.step(x_observation)
        cur_state_pred_ = np.concatenate((cur_state_pred.reshape(1, -1), np.zeros((1, X_control.shape[1]))), axis=1)
        cur_state_pred = state_transition_model._scaler.inverse_transform(cur_state_pred_).flatten()[:cur_state_pred.shape[0]]  # Undo scaling
        
        cov_state_pred = np.diag(cov_state_pred)
        cov_state_pred_ = np.concatenate((cov_state_pred.reshape(1, -1), np.zeros((1, X_control.shape[1]))), axis=1)
        cov_state_pred = state_transition_model._scaler.inverse_transform(cov_state_pred_).flatten()[:cur_state_pred.shape[0]]
        std_state_pred = np.sqrt(cov_state_pred)

        # Evaluate (only Cl concentration states)
        cur_state = X_cur_state[i, :]
        cur_state_ = np.concatenate((cur_state.reshape(1, -1), np.zeros((1, X_control.shape[1]))), axis=1)
        cur_state = state_transition_model._scaler.inverse_transform(cur_state_).flatten()[:cur_state.shape[0]]  # Undo scaling

        cl_pred.append(cur_state_pred[:n_cl_items].reshape(1, -1))
        cl_true.append(cur_state[:n_cl_items].reshape(1, -1))
        cl_pred_std.append(std_state_pred[:n_cl_items])
        avg_cl_scores.append(np.median(np.abs(cur_state_pred[:n_cl_items] - cur_state[:n_cl_items])))

        """
        t = mape(cur_state_pred[:n_cl_items], cur_state[:n_cl_items])

        t_ = [] # Compute percentage deviation
        for idx in range(n_cl_items):
            y_true = cur_state[idx]
            y_pred = cur_state_pred[idx]

            t_.append(np.abs((y_pred - y_true) / y_true))
        #avg_cl_scores.append(np.median(t_))
        """

        # Evaluate (only flows)
        flows_pred.append(cur_state_pred[n_cl_items:].reshape(1, -1))
        flows_true.append(cur_state[n_cl_items:].reshape(1, -1))
        avg_flow_scores.append(np.median(np.abs(cur_state_pred[n_cl_items:] - cur_state[n_cl_items:])))

    return avg_cl_scores, cl_pred, cl_true, cl_pred_std, avg_flow_scores, flows_pred, flows_true


def run_exp(n_sensors_range: list[int], n_iters: int, **kwds):
    cl_pred_score_avg = []
    cl_pred_score_std = []

    for n_sensors in n_sensors_range:
        scores = []
        for _ in range(n_iters):
            avg_cl_scores, _, _, cl_pred_std, _, _, _ = \
                run_state_estimation(n_node_quality_sensors=n_sensors, n_link_sensors=n_sensors, **kwds)
            scores.append(avg_cl_scores)

        cl_pred_score_avg.append(np.mean(scores))
        cl_pred_score_std.append(np.std(scores))

    return cl_pred_score_avg, cl_pred_score_std


if __name__ == "__main__":
    # CY-DBP
    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
                n_iters=30, net_desc="CY-DBP",
                scada_file_in=os.path.join("data", "cydbp_randDemand=True_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "cydbp_randDemand=True_test.npz"),
                state_transition_model_file_in=os.path.join("data", "cydbp_randDemand=True_surrogate.pt"))
    np.savez("exp-results-cydbp.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
                n_iters=30, net_desc="CY-DBP",
                scada_file_in=os.path.join("data", "cydbp_randDemand=False_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "cydbp_randDemand=False_test.npz"),
                state_transition_model_file_in=os.path.join("data", "cydbp_randDemand=True_surrogate.pt"))
    np.savez("exp-results-cydbp-randDemand=False.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

    # Hanoi
    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                n_iters=30, net_desc="Hanoi",
                scada_file_in=os.path.join("data", "hanoi_randDemand=False_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "hanoi_randDemand=False_test.npz"),
                state_transition_model_file_in=os.path.join("data", "hanoi_randDemand=True_surrogate.pt"))
    np.savez("exp-results-hanoi_randDemand=False.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

    # Net1
    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                n_iters=30, net_desc="Hanoi",
                scada_file_in=os.path.join("data", "hanoi_randDemand=True_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "hanoi_randDemand=True_test.npz"),
                state_transition_model_file_in=os.path.join("data", "hanoi_randDemand=True_surrogate.pt"))
    np.savez("exp-results-hanoi.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9],
                n_iters=30, net_desc="Net1",
                scada_file_in=os.path.join("data", "net1_randDemand=True_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "net1_randDemand=True_test.npz"),
                state_transition_model_file_in=os.path.join("data", "net1_randDemand=True_surrogate.pt"))
    np.savez("exp-results-net1.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

    r = run_exp(n_sensors_range=[2, 3, 4, 5, 6, 7, 8, 9],
                n_iters=30, net_desc="Net1",
                scada_file_in=os.path.join("data", "net1_randDemand=False_test.epytflow_scada_data"),
                control_actions_file_in=os.path.join("data", "net1_randDemand=False_test.npz"),
                state_transition_model_file_in=os.path.join("data", "net1_randDemand=True_surrogate.pt"))
    np.savez("exp-results-net1_randDemand=False.npz", cl_pred_score_avg=r[0], cl_pred_score_std=r[1])
    print(r)

