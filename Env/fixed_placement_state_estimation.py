import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from epyt_flow.simulation import ScadaData
from epyt_control.signal_processing.state_estimation import (
    TimeVaryingExtendedKalmanFilter,
)

from run_exp_state_estimation import (
    get_state_transition_model,
)


def create_fixed_sensor_placement(
    node_indices,
    link_indices,
    n_nodes,
    n_links,
    state_dim,
):
    """
    Same measurement matrix layout as
    create_random_sensor_placement() from the reference code,
    but with explicitly supplied sensor indices.
    """

    node_indices = sorted(node_indices)
    link_indices = sorted(link_indices)

    M = np.zeros(
        (
            len(node_indices)
            + 2 * len(link_indices),
            state_dim,
        )
    )

    flows_idx = []

    i = 0

    # Node chlorine
    for idx in node_indices:
        M[i, idx] = 1
        i += 1

    # Link chlorine
    for idx in link_indices:
        M[
            i,
            n_nodes + idx,
        ] = 1
        i += 1

    # Flow at selected links
    for idx in link_indices:
        j = (
            n_nodes
            + n_links
            + idx
        )

        flows_idx.append(j)
        M[i, j] = 1
        i += 1

    return M, flows_idx


def run_state_estimation_fixed(
    net_desc,
    scada_file_in,
    control_actions_file_in,
    state_transition_model_file_in,
    node_indices,
    link_indices,
):
    """
    Adapted from run_exp_state_estimation.run_state_estimation.

    Only conceptual modification:
    fixed sensor indices are used instead of random placement.
    """

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    scada_data = ScadaData.load_from_file(
        scada_file_in
    )

    control_actions = np.load(
        control_actions_file_in
    )["control_actions"]

    X_flows = scada_data.get_data_flows()
    X_nodes_quality = (
        scada_data.get_data_nodes_quality()
    )
    X_links_quality = (
        scada_data.get_data_links_quality()
    )

    n_time_steps = X_flows.shape[0]

    next_flow = X_flows[1:, :]

    cur_node_quality = (
        X_nodes_quality[
            :n_time_steps - 1,
            :
        ]
    )

    cur_link_quality = (
        X_links_quality[
            :n_time_steps - 1,
            :
        ]
    )

    next_state = np.concatenate(
        (
            X_nodes_quality[1:, :],
            X_links_quality[1:, :],
        ),
        axis=1,
    )

    X_cur_state = np.concatenate(
        (
            cur_node_quality,
            cur_link_quality,
            next_flow,
        ),
        axis=1,
    )

    X_control = control_actions[
        :n_time_steps - 1,
        :
    ]

    state_dim = X_cur_state.shape[1]
    n_cl_items = next_state.shape[1]

    n_nodes = cur_node_quality.shape[1]
    n_links = cur_link_quality.shape[1]

    # --------------------------------------------------------
    # Surrogate
    # --------------------------------------------------------

    state_transition_model = (
        get_state_transition_model(
            net_desc,
            state_transition_model_file_in,
        )
    )

    state_transition_model.n_missing_flows = (
        next_state.shape[1]
    )

    state_transition_model._normalize_input_output = (
        False
    )

    X_cur_state_ = np.concatenate(
        (
            X_cur_state,
            X_control,
        ),
        axis=1,
    )

    X_cur_state = (
        state_transition_model
        ._scaler
        .transform(X_cur_state_)[
            :,
            :X_cur_state.shape[1],
        ]
    )

    # --------------------------------------------------------
    # ONLY CHANGED PART:
    # fixed instead of random sensor placement
    # --------------------------------------------------------

    M, flows_idx = (
        create_fixed_sensor_placement(
            node_indices=node_indices,
            link_indices=link_indices,
            n_nodes=n_nodes,
            n_links=n_links,
            state_dim=state_dim,
        )
    )

    obs_dim = M.shape[0]

    # --------------------------------------------------------
    # EKF -- same logic as reference implementation
    # --------------------------------------------------------

    measurement_func = (
        lambda x:
        np.dot(
            M,
            x.flatten(),
        )
    )

    measurement_func_grad = (
        lambda _:
        M
    )

    def get_measurement_func(t):
        return measurement_func

    def get_measurement_func_grad(t):
        return measurement_func_grad

    def get_control_signal(t):
        return X_control[
            t + 1,
            :
        ].reshape(
            1,
            -1,
        )

    def get_state_transition_func(t):
        x_control = get_control_signal(t)

        return lambda x: (
            state_transition_model.predict(
                x.reshape(1, -1),
                x_control,
            ).flatten()
        )

    def get_state_transition_func_grad(t):
        x_control = get_control_signal(t)

        def get_jac(x_cur_state):
            jac = (
                state_transition_model
                .compute_jacobian(
                    x_cur_state.reshape(
                        1,
                        -1,
                    ),
                    x_control,
                )
            )

            jac = jac.reshape(
                jac.shape[1],
                jac.shape[3],
            )

            jac = jac[
                :,
                :state_dim,
            ]

            return jac

        return get_jac

    my_filter = (
        TimeVaryingExtendedKalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            init_state=X_cur_state[0, :],
            get_state_transition_func=(
                get_state_transition_func
            ),
            get_state_transition_func_grad=(
                get_state_transition_func_grad
            ),
            get_measurement_func=(
                get_measurement_func
            ),
            get_measurement_func_grad=(
                get_measurement_func_grad
            ),
        )
    )

    # --------------------------------------------------------
    # Apply EKF
    # --------------------------------------------------------

    cl_pred = []
    cl_true = []

    for i in range(
        1,
        X_cur_state.shape[0],
    ):

        cur_state = X_cur_state[
            i,
            :
        ]

        # Same direct flow injection as reference implementation.
        for idx in flows_idx:
            my_filter._x[idx] = (
                cur_state[idx]
            )

        x_observation = (
            measurement_func(
                X_cur_state[i, :]
            )
        )

        (
            cur_state_pred,
            _,
        ) = my_filter.step(
            x_observation
        )

        # Undo scaling of prediction
        cur_state_pred_ = np.concatenate(
            (
                cur_state_pred.reshape(
                    1,
                    -1,
                ),
                np.zeros(
                    (
                        1,
                        X_control.shape[1],
                    )
                ),
            ),
            axis=1,
        )

        cur_state_pred = (
            state_transition_model
            ._scaler
            .inverse_transform(
                cur_state_pred_
            )
            .flatten()[
                :cur_state_pred.shape[0]
            ]
        )

        # Undo scaling of ground truth
        cur_state = X_cur_state[
            i,
            :
        ]

        cur_state_ = np.concatenate(
            (
                cur_state.reshape(
                    1,
                    -1,
                ),
                np.zeros(
                    (
                        1,
                        X_control.shape[1],
                    )
                ),
            ),
            axis=1,
        )

        cur_state = (
            state_transition_model
            ._scaler
            .inverse_transform(
                cur_state_
            )
            .flatten()[
                :cur_state.shape[0]
            ]
        )

        cl_pred.append(
            cur_state_pred[
                :n_cl_items
            ].reshape(
                1,
                -1,
            )
        )

        cl_true.append(
            cur_state[
                :n_cl_items
            ].reshape(
                1,
                -1,
            )
        )

    return cl_pred, cl_true