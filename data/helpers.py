import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex, LinearSegmentedColormap
from sklearn.manifold import TSNE
import torch

import model
import configs

# manually order objects by similarity
SIM_OBJECTS_LIST = ['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                    'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                    'wooden-button', 'glass-bead', 'plastic-bead', 'metal-nut-bolt']
SORTED_DATA_OBJ_LIST = sorted(SIM_OBJECTS_LIST)  # for input data from data files, the labels were created in this order


def generate_colors(n_colors):
    cmap = plt.cm.hsv  # Use HSV colormap for hue-based colors
    colors = [cmap(i / n_colors) for i in range(n_colors)]
    return colors


def sanity_check_data_labels(data_dict: dict):
    """ check if the data's labels were created in the order of the sorted object list: SORTED_DATA_OBJ_LIST"""
    mismatch = False
    for behavior in data_dict.keys():
        for tool in data_dict[behavior].keys():
            for modality in data_dict[behavior][tool].keys():
                for obj in data_dict[behavior][tool][modality].keys():
                    trail_batch = data_dict[behavior][tool][modality][obj]
                    trail_batch_label = np.squeeze(trail_batch['Y'])
                    for trial_num, label in enumerate(trail_batch_label):
                        try:
                            assert obj == SORTED_DATA_OBJ_LIST[label]
                        except AssertionError:
                            mismatch = True
                            print(f"object does not match label: {behavior}_{tool}_{modality}_{obj}-trial-{trial_num}: "
                                  f"{label} instead of {SORTED_DATA_OBJ_LIST.index(obj)}")
    if mismatch:
        raise AssertionError


def stack_input_data(X_array, Y_array, behavior_list, tool_list, obj_list, data):
    """stack new data to X_array, Y_array"""
    X_array = [X_array] if len(X_array) != 0 else []
    Y_array = [Y_array] if len(Y_array) != 0 else []
    meta_data = {b: {t: {} for t in tool_list} for b in behavior_list}
    for b_idx, b in enumerate(behavior_list):
        for t_idx, t in enumerate(tool_list):
            for o_idx, o in enumerate(obj_list):
                X_array.append(data[b][t]['audio'][o]['X'])
                Y_array.append(data[b][t]['audio'][o]['Y'])
                meta_data[b][t][o] = len(data[b][t]['audio'][o]['Y'])
    logging.debug(f"data input meta_data: {meta_data}")
    return np.vstack(X_array), np.vstack(Y_array)


def viz_input_data(data, loss_func_name: str, behavior_list, source_tool_list, target_tool_list, old_object_list,
                   new_object_list, shared_only: bool, test_only: bool, plot_title="Original Data Space"):
    assert ~(shared_only and test_only)  # can't both be True
    assert len(target_tool_list) == 1  # one target tool only
    if shared_only:
        logging.info("visualize the SHARE object set in input data space...")
        all_obj_list = old_object_list
        new_object_list = []
    elif test_only:
        logging.info("visualize the TEST object set in input data space...")
        all_obj_list = new_object_list
        old_object_list = []
    else:
        logging.info("visualize the ALL object set in input data space...")
        all_obj_list = old_object_list + new_object_list

    X_array, Y_array = stack_input_data([], [], behavior_list, source_tool_list, all_obj_list, data=data)
    len_source = len(behavior_list) * len(source_tool_list) * len(all_obj_list) * 10
    len_target_shared = len(behavior_list) * len(old_object_list) * 10
    len_target_test = len(behavior_list) * len(new_object_list) * 10
    if not test_only:
        X_array, Y_array = stack_input_data(X_array, Y_array, behavior_list, target_tool_list, old_object_list,
                                            data=data)
        if shared_only:
            len_target_test = 0
    if not shared_only:
        X_array, Y_array = stack_input_data(X_array, Y_array, behavior_list, target_tool_list, new_object_list,
                                            data=data)
        if test_only:
            len_target_shared = 0
    # map original labels to new labels (based on SIM_OBJECTS_LIST) to match the color bar build on SIM_OBJECTS_LIST
    Y_objects = [SORTED_DATA_OBJ_LIST[label] for label in np.squeeze(Y_array)]
    Y_curr_labels = np.array([all_obj_list.index(obj) for obj in Y_objects])  # reassign labels based on all_obj_list
    viz_shared_latent_space(loss_func=loss_func_name, obj_list=all_obj_list,
                            embeds=X_array, labels=Y_curr_labels, save_fig=False,
                            len_list=[len_source, len_target_shared, len_target_test], title=plot_title)


def viz_embeddings(transfer_class, loss_func, viz_objects: list, input_dim,
                   source_tool_list, target_tool_list, modality_list, trail_list, behavior_list,
                   old_object_list, new_object_list, encoder_state_dict_loc: str = './saved_model/encoder/'):
    encoder_pt_name = f"myencoder_{loss_func}.pt"
    Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                            hidden_size=configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(
        torch.load(encoder_state_dict_loc + encoder_pt_name, map_location=torch.device(configs.device)))

    def vis_by_condition(curr_old_object_list, curr_new_object_list, curr_new_obj_only, curr_train_obj_only):
        all_obj_list = curr_old_object_list + curr_new_object_list
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder, new_obj_only=curr_new_obj_only, train_obj_only=curr_train_obj_only, behavior_list=behavior_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            modality_list=modality_list, old_object_list=curr_old_object_list, new_object_list=curr_new_object_list,
            trail_list=trail_list)
        viz_shared_latent_space(
            loss_func=loss_func, obj_list=all_obj_list, embeds=all_embeds, labels=all_labels,
            len_list=[source_len, target_len, target_test_len])

    if "shared" in viz_objects:
        logging.info("visualize the SHARE object set in shared latent space...")
        curr_new_object_list = []
        curr_new_obj_only, curr_train_obj_only = False, True
        vis_by_condition(old_object_list, curr_new_object_list, curr_new_obj_only, curr_train_obj_only)
    if "test" in viz_objects:
        logging.info("visualize the TEST object set in shared latent space...")
        curr_old_object_list = []
        curr_new_obj_only, curr_train_obj_only = True, False
        vis_by_condition(curr_old_object_list, new_object_list, curr_new_obj_only, curr_train_obj_only)
    if "all" in viz_objects:
        logging.info("visualize the ALL objects in shared latent space...")
        curr_new_obj_only, curr_train_obj_only = False, False
        vis_by_condition(old_object_list, new_object_list, curr_new_obj_only, curr_train_obj_only)


def viz_shared_latent_space(loss_func: str, obj_list: list, embeds: np.ndarray,
                            labels: np.ndarray, len_list: list, save_fig: bool = True, title='') -> None:
    """
    !!!make sure that labels were created following the index of obj_list!!!

    :param loss_func: name for the loss function
    :param obj_list: set of all object names, in the order of old + new
    :param embeds:  array of all 1D embeddings from source (all and/or new obj), target (old obj), and target test (new obj),
                        shape=(sum(len_list), len_1D_emb)
    :param labels:
    :param len_list: length of data from each tool&(object set) combo:
                        [n_emb_source_tool_source_objects, n_emb_target_tool_old_obj, n_emb_target_tool_new_obj]:
                        (n_source_tools * n_source_objects * n_trials, n_old_objects * n_trials, n_new_objects * n*n_trials)
    :param save_fig: save the fig or not
    :param title: customized title section, following "T-SNE Visualization of Embeddings in "
    :return: None
    """
    logging.debug(f"➡️ viz_shared_latent_space..")
    source_len, target_len, target_test_len = len_list
    labels = np.squeeze(labels)
    # Create tool labels: 0 for source, 1 for target, 2 for target test
    tool_labels = np.array([0] * source_len + [1] * target_len + [2] * target_test_len)

    logging.debug(f"obj_list: {obj_list}")
    logging.debug(f"source_len, target_len, target_test_len: {source_len, target_len, target_test_len}")
    logging.debug(f"embeds shape: {embeds.shape}")
    logging.debug(f"labels: \n      {labels}")

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
    embeds_2d = tsne.fit_transform(embeds)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    markers = ['o', '^', 'x']  # Markers for tools. Circle for source, triangle for target, x for target test

    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    subset_obj_idx = [SIM_OBJECTS_LIST.index(obj) for obj in obj_list]
    logging.debug(f"subset_obj_idx: {subset_obj_idx}, unique labels: {np.unique(labels)}")
    sim_colors = plt.cm.tab20b(np.linspace(0, 1, len(SIM_OBJECTS_LIST)))  # fix colors for all 15 objects
    subset_colors = sim_colors[sorted(subset_obj_idx)]  # select the subset colors, value has to be
    cmap = ListedColormap(subset_colors)  # Discrete colormap
    bounds = list(range(len(obj_list) + 1))  # Boundaries for discrete colors
    norm = BoundaryNorm(bounds, len(obj_list))  # Normalize to discrete boundaries

    # map current label (ordered by obj_list) to align with color (ordered by SIM_OBJECTS_LIST)
    sorted_indices = sorted(enumerate(subset_obj_idx), key=lambda x: x[1])
    sorted_unique_orig_labels = sorted(np.unique(labels))
    # Step 1: Remove duplicates from subset_obj_idx while preserving the order
    unique_subset_idx = list(dict.fromkeys(subset_obj_idx))  # [9, 2, 14, 13, 8]

    # Step 2: Create a mapping from original labels to indices in the sorted subset
    sim_obj_idx = [SIM_OBJECTS_LIST.index(obj_list[l]) for l in labels]
    label_to_subset_idx = {label: sim_obj_idx[i] for i, label in enumerate(labels)}  # Map labels to subset indices
    sorted_unique_idx = sorted(unique_subset_idx)  # Sort the unique subset indices
    index_mapping = {idx: i for i, idx in enumerate(sorted_unique_idx)}  # Map sorted indices to new positions

    # Step 3: Map original labels to new positions
    mapped_labels = np.array([index_mapping[label_to_subset_idx[label]] for label in labels])

    # here, for the color bar object reference only,
    # in case the last batch of objects for plt are a subset of obj_list
    scatter = plt.scatter(
        embeds_2d[:, 0], embeds_2d[:, 1],
        c=mapped_labels,
        cmap=cmap,
        norm=norm,
        s=0 # Use invisible points
    )

    for tool_label, marker in enumerate(markers):
        mask = tool_labels == tool_label
        masked_labels = labels[mask]
        if len(masked_labels) == 0:
            continue
        plt.scatter(
            embeds_2d[mask, 0], embeds_2d[mask, 1],
            c=mapped_labels[mask],
            cmap=cmap,  # Use discrete colormap
            norm=norm,
            marker=marker,
            s=80,
            alpha=0.9,
            label=f"{['Source Tool(All)', 'Target Tool(Train)', 'Target Tool(Test)'][tool_label]}"
        )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(obj_list)))
    cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])  # colors are sorted by sim object, so should the object names
    cbar.set_label("Objects", rotation=270, labelpad=20)

    plt.legend(title="Tool Type")
    save_name = title if title else f"shared_space-{loss_func} loss"
    plt.title(f"T-SNE Visualization of Embeddings in {save_name}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    if save_fig:
        plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

# def viz_shared_latent_space(loss_func: str, all_obj_list: list, all_embeds: np.ndarray,
#                             all_labels: np.ndarray, len_list: list, save_fig: bool = True, title='') -> None:
#     """
#
#     :param loss_func: name for the loss function
#     :param all_obj_list: set of all object names, in the order of old + new
#     :param all_embeds:  array of all 1D embeddings from source (all and/or new obj), target (old obj), and target test (new obj),
#                         shape=(sum(len_list), len_1D_emb)
#     :param all_labels:
#     :param len_list: length of data from each tool&(object set) combo:
#                         [n_emb_source_tool_source_objects, n_emb_target_tool_old_obj, n_emb_target_tool_new_obj]:
#                         (n_source_tools * n_source_objects * n_trials, n_old_objects * n_trials, n_new_objects * n*n_trials)
#     :param save_fig: save the fig or not
#     :param title: customized title section, following "T-SNE Visualization of Embeddings in "
#     :return: None
#     """
#     logging.debug(f"➡️ viz_shared_latent_space..")
#     source_len, target_len, target_test_len = len_list
#     all_labels = np.squeeze(all_labels, axis=None)
#     # Create tool labels: 0 for source, 1 for target, 2 for target test
#     tool_labels = np.array([0] * source_len + [1] * target_len + [2] * target_test_len)
#
#     logging.debug(f"all_obj_list: {all_obj_list}")
#     logging.debug(f"source_len, target_len, target_test_len: {source_len, target_len, target_test_len}")
#     logging.debug(f"all_embeds shape: {all_embeds.shape}")
#     logging.debug(f"all_labels: \n      {all_labels}")
#
#     # Step 1: Dimensionality reduction using t-SNE
#     tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
#     embeds_2d = tsne.fit_transform(all_embeds)
#
#     plt.figure(figsize=(8, 6))
#     plt.rcParams['font.size'] = 12
#     markers = ['o', '^', 'x']  # Markers for tools. Circle for source, triangle for target, x for target test
#     colors = plt.cm.tab20(np.linspace(0, 1, len(all_obj_list)))
#     cmap = ListedColormap(colors)  # Discrete colormap
#     bounds = list(range(len(all_obj_list) + 1))  # Boundaries for discrete colors
#     norm = BoundaryNorm(bounds, cmap.N)  # Normalize to discrete boundaries
#
#     scatter = plt.scatter(
#         embeds_2d[:, 0], embeds_2d[:, 1],
#         c=all_labels,
#         cmap=cmap,  # Use discrete colormap
#         norm=norm,  # Apply discrete normalization
#         s=0  # Use invisible points for the color bar reference
#     )
#
#     for tool_label, marker in enumerate(markers):
#         mask = tool_labels == tool_label
#         plt.scatter(
#             embeds_2d[mask, 0], embeds_2d[mask, 1],
#             c=all_labels[mask],
#             # without discrete normalization, cmap will align labels differently when they don't have the same range (i.e. labels [0, 3] range is not 15)
#             cmap=cmap,  # Use discrete colormap
#             norm=norm,  # Apply discrete normalization
#             marker=marker,
#             s=50,
#             alpha=0.7,
#             label=f"{['Source Tool(All)', 'Target Tool(Train)', 'Target Tool(Test)'][tool_label]}"
#         )
#
#     # Step 3: Add legend for tools (shapes)
#     plt.legend(title="Tool Type")
#
#     # Step 4: Add color bar for objects
#     cbar = plt.colorbar(scatter, ticks=range(len(all_obj_list)))
#     cbar.ax.set_yticklabels(all_obj_list)
#     cbar.set_label("Objects", rotation=270, labelpad=20)
#
#     # Plot details
#     if title:
#         save_name = title
#     else:
#         save_name = f"shared_space-{loss_func} loss"
#     plt.title(f"T-SNE Visualization of Embeddings in {save_name}")
#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")
#     plt.grid(True)
#
#     if save_fig:
#         plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
#     plt.show()
#     plt.close()
