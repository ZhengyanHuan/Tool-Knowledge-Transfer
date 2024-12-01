import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex, LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
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


def viz_input_data(data, shared_only: bool, test_only: bool, loss_func_name: str = configs.loss_func,
                   behavior_list=configs.behavior_list, source_tool_list=configs.source_tool_list,
                   target_tool_list=configs.target_tool_list, old_object_list=configs.old_object_list,
                   new_object_list=configs.new_object_list, plot_title="Original Data Space"):
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
            plot_title += "(Shared Object Only)"
    if not shared_only:
        X_array, Y_array = stack_input_data(X_array, Y_array, behavior_list, target_tool_list, new_object_list,
                                            data=data)
        if test_only:
            len_target_shared = 0
            plot_title += "(Test Object Only)"
    # map original labels to new labels (based on SIM_OBJECTS_LIST) to match the color bar build on SIM_OBJECTS_LIST
    Y_objects = [SORTED_DATA_OBJ_LIST[label] for label in np.squeeze(Y_array)]
    Y_curr_labels = np.array([all_obj_list.index(obj) for obj in Y_objects])  # reassign labels based on all_obj_list

    subtitle = f"target tool: {target_tool_list}, source tool: {source_tool_list}"
    viz_embeddings(loss_func=loss_func_name, obj_list=all_obj_list,
                   embeds=X_array, labels=Y_curr_labels, save_fig=True,
                   len_list=[len_source, len_target_shared, len_target_test], title=plot_title, subtitle=subtitle)


def viz_embeddings_by_object_set(viz_objects: list, transfer_class, input_dim: int, loss_func=configs.loss_func,
                                 source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
                                 modality_list=configs.modality_list, trail_list=configs.trail_list,
                                 behavior_list=configs.behavior_list,
                                 old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                                 encoder_state_dict_loc: str = './saved_model/encoder/'):
    encoder_pt_name = f"myencoder_{loss_func}.pt"
    Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                            hidden_size=configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(
        torch.load(encoder_state_dict_loc + encoder_pt_name, map_location=torch.device(configs.device)))

    def vis_by_condition(curr_old_object_list, curr_new_object_list, curr_new_obj_only, curr_train_obj_only, title):
        all_obj_list = curr_old_object_list + curr_new_object_list
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder, new_obj_only=curr_new_obj_only, train_obj_only=curr_train_obj_only, behavior_list=behavior_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            modality_list=modality_list, old_object_list=curr_old_object_list, new_object_list=curr_new_object_list,
            trail_list=trail_list)
        subtitle = f"target tool: {target_tool_list}, source tool: {source_tool_list}"
        viz_embeddings(
            loss_func=loss_func, obj_list=all_obj_list, embeds=all_embeds, labels=all_labels,
            len_list=[source_len, target_len, target_test_len], save_fig=True, title=title, subtitle=subtitle)

    if "shared" in viz_objects:
        logging.info("visualize the SHARE object set in shared latent space...")
        curr_new_object_list = []
        curr_new_obj_only, curr_train_obj_only = False, True
        vis_by_condition(old_object_list, curr_new_object_list, curr_new_obj_only, curr_train_obj_only,
                         title=f"shared_space-{loss_func}(Shared Object Only)")
    if "test" in viz_objects:
        logging.info("visualize the TEST object set in shared latent space...")
        curr_old_object_list = []
        curr_new_obj_only, curr_train_obj_only = True, False
        vis_by_condition(curr_old_object_list, new_object_list, curr_new_obj_only, curr_train_obj_only,
                         title=f"shared_space-{loss_func}(Test Object Only)")
    if "all" in viz_objects:
        logging.info("visualize the ALL objects in shared latent space...")
        curr_new_obj_only, curr_train_obj_only = False, False
        vis_by_condition(old_object_list, new_object_list, curr_new_obj_only, curr_train_obj_only,
                         title=f"shared_space-{loss_func}")


def map_objects_to_colors(obj_list, labels):
    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    subset_obj_idx = [SIM_OBJECTS_LIST.index(obj) for obj in obj_list]
    logging.debug(f"subset_obj_idx: {subset_obj_idx}, unique labels: {np.unique(labels)}")
    sim_colors = plt.colormaps['tab20b'](np.linspace(0, 1, len(SIM_OBJECTS_LIST)))  # fix colors for all 15 objects
    subset_colors = sim_colors[sorted(subset_obj_idx)]  # select the subset colors, value has to be
    cmap = ListedColormap(subset_colors)  # Discrete colormap
    bounds = list(range(len(obj_list) + 1))  # Boundaries for discrete colors
    norm = BoundaryNorm(bounds, len(obj_list))  # Normalize to discrete boundaries

    # map current label (ordered by obj_list) to align with color (ordered by SIM_OBJECTS_LIST)
    # Step 1: Remove duplicates from subset_obj_idx while preserving the order
    unique_subset_idx = list(dict.fromkeys(subset_obj_idx))  # [9, 2, 14, 13, 8]

    # Step 2: Create a mapping from original labels to indices in the sorted subset
    sim_obj_idx = [SIM_OBJECTS_LIST.index(obj_list[int(l)]) for l in labels]
    label_to_subset_idx = {label: sim_obj_idx[i] for i, label in enumerate(labels)}  # Map labels to subset indices
    sorted_unique_idx = sorted(unique_subset_idx)  # Sort the unique subset indices
    index_mapping = {idx: i for i, idx in enumerate(sorted_unique_idx)}  # Map sorted indices to new positions

    # Step 3: Map original labels to new positions
    mapped_labels = np.array([index_mapping[label_to_subset_idx[label]] for label in labels])
    return mapped_labels, cmap, norm, subset_obj_idx, label_to_subset_idx


def get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx):
    cbar_labels = np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)]
    reversed_mp = {sim_idx: label for label, sim_idx in label_to_subset_idx.items()}
    original_labels = [reversed_mp[sim_idx] for sim_idx in sorted(subset_obj_idx)]
    tick_labels = [f"{cbar_label}\n(label {orig_label})" for cbar_label, orig_label in
                   zip(cbar_labels, original_labels)]
    return tick_labels


def viz_test_objects_embedding(transfer_class, Encoder, Classifier, pred_label_target):
    # if embeddings need feature reduction to 2D, show the actual predictions for reference
    if configs.encoder_output_dim > 2:
        *_, pred_label_source = transfer_class.eval(Encoder=Encoder, Classifier=Classifier,  # evaluate source tool
                                                    tool_list=configs.source_tool_list, return_pred=True)
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder=Encoder, new_obj_only=True)
        labels = np.concatenate([pred_label_source.cpu().detach().numpy(), pred_label_target.cpu().detach().numpy()],
                                axis=0)
        viz_embeddings(obj_list=configs.new_object_list, embeds=all_embeds, labels=labels,
                       len_list=[source_len, target_len, target_test_len], show_curr_label=True,
                       subtitle=f"Test Predictions. Target {configs.target_tool_list} \n Source: {configs.source_tool_list}")

    # visualize the data in 2 D space. The original labels are preserved because there will be colored background for predicted labels
    # If the embedding is 2D, use the trained Classifier for decision boundaries, the background color will match the actual predictions
    # if > 2, the boundaries are from a logistic regression clf trained on t-sne reduced 2D space,
    #   the background does not reflect actual predictions, it approximates what a linear classifier does on full-sized embeddings.
    viz_classifier_boundary_on_2d_embeddings(transfer_class, Encoder, Classifier)


def viz_embeddings(embeds: np.ndarray, labels: np.ndarray, len_list: list,
                   loss_func: str = configs.loss_func,
                   obj_list: list = configs.old_object_list + configs.new_object_list,
                   save_fig: bool = True, title='', subtitle='', show_curr_label=False) -> None:
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
    if embeds.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
        embeds_2d = tsne.fit_transform(embeds)
    else:
        embeds_2d = embeds

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    markers = ['o', '^', 's']  # Markers for tools. Circle for source, triangle for target, x for target test

    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    mapped_labels, cmap, norm, subset_obj_idx, label_to_subset_idx = map_objects_to_colors(obj_list, labels)

    # for the color bar reference only, in case the last batch of objects for plt are a subset of obj_list
    scatter = plt.scatter(
        embeds_2d[:, 0], embeds_2d[:, 1],
        c=mapped_labels,
        cmap=cmap,
        norm=norm,
        s=0  # Use invisible points
    )

    for tool_label, marker in enumerate(markers):
        mask = tool_labels == tool_label
        masked_labels = labels[mask]
        if len(masked_labels) == 0:
            continue
        edge_color = 'black' if tool_label == 2 else None  # add edge to test tool marker
        plt.scatter(
            embeds_2d[mask, 0], embeds_2d[mask, 1],
            c=mapped_labels[mask],
            cmap=cmap,  # Use discrete colormap
            norm=norm,
            marker=marker,
            edgecolor=edge_color,
            s=80,
            alpha=0.9,
            label=f"{['Source Tool(All)', 'Target Tool(Train)', 'Target Tool(Test)'][tool_label]}"
        )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(obj_list)))
    if show_curr_label:
        cbar.ax.set_yticklabels(get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx))
    else:
        cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(
            subset_obj_idx)])  # colors are sorted by sim object, so should the object names
    cbar.set_label("Objects", rotation=270, labelpad=20)

    plt.legend(title="Tool Type")
    save_name = title if title else f"shared_space-{loss_func} loss"
    viz_discpt = "T-SNE " if embeds.shape[1] > 2 else ""
    plt.title(f"{viz_discpt}Visualization of Embeddings in {save_name} \n {subtitle}")
    plt.xlabel(f"{viz_discpt}Dimension 1")
    plt.ylabel(f"{viz_discpt}Dimension 2")
    plt.grid(True)

    if save_fig:
        plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
    plt.show()
    plt.close()


def viz_classifier_boundary_on_2d_embeddings(transfer_class, Encoder, Classifier):
    object_list = configs.new_object_list
    # Step 1: Generate embedded data
    all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(Encoder=Encoder,
                                                                                                     new_obj_only=True)
    source_emb = all_embeds[:source_len]
    source_y = np.squeeze(all_labels[:source_len])
    target_emb = all_embeds[-target_test_len:]
    target_y = np.squeeze(all_labels[-target_test_len:])

    # Step 2:
    if all_embeds.shape[1] > 2:  # Apply T-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
        all_embeddings_2d = tsne.fit_transform(all_embeds)
        embeddings_2d = all_embeddings_2d[:source_len]
        new_embeddings_2d = all_embeddings_2d[-target_test_len:]
    elif all_embeds.shape[1] == 2:
        embeddings_2d = source_emb
        new_embeddings_2d = target_emb
        all_embeddings_2d = all_embeds
    else:
        raise Exception(f"embedding shape is not correct: {all_embeds.shape}")

    # Step 4: Create a grid for decision boundary
    x_min, x_max = all_embeddings_2d[:, 0].min() - 1, all_embeddings_2d[:, 0].max() + 1
    y_min, y_max = all_embeddings_2d[:, 1].min() - 1, all_embeddings_2d[:, 1].max() + 1
    n_points = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # (n_points*n_points, 2)
    # Step 2: Train a classifier in the 2D T-SNE space
    if all_embeds.shape[1] == 2:  # use the trained classifier directly because input shape matches
        grid_proba = Classifier(torch.tensor(grid_points, dtype=torch.float32, device=configs.device))
        grid_proba = grid_proba.cpu().detach().numpy()  # (n_points*n_points, C)
    else:  # use LogisticRegression to classify data on the t-sne transformed space (approximate the decision boundaries of the Classifier)
        secondary_classifier = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
        secondary_classifier.fit(embeddings_2d, source_y)
        grid_proba = secondary_classifier.predict_proba(grid_points)  # Shape: [n_grid_points, C]

    grid_classes = np.argmax(grid_proba, axis=1)  # (n_points*n_points, )

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12

    # Step 5: Plot the decision boundary
    levels = np.arange(-1, len(object_list))  # strangely it needs the lowest level as -1 to show all colors!!!
    grid_classes, cmap, norm, subset_obj_idx, label_to_subset_idx = map_objects_to_colors(object_list, grid_classes)
    plt.contourf(xx, yy, grid_classes.reshape(xx.shape), cmap=cmap, levels=levels,
                 alpha=0.3)  # contourf flips the array upside down

    # Step 6: Overlay the original and test embeddings
    pred_label_source, *_ = map_objects_to_colors(object_list, source_y)
    pred_label_target, *_ = map_objects_to_colors(object_list, target_y)
    scatter_original = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=pred_label_source, cmap=cmap, norm=norm, s=50, alpha=0.8, label="Original Embeddings"
    )
    plt.scatter(
        new_embeddings_2d[:, 0], new_embeddings_2d[:, 1],
        c=pred_label_target, cmap=cmap, norm=norm, edgecolor='k', s=80, alpha=1.0, marker='s', label="Test Embeddings"
    )

    # Add a color bar and legend
    cbar = plt.colorbar(scatter_original, ticks=np.arange(len(object_list)))
    cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])
    cbar.set_label("Classes", rotation=270, labelpad=20)

    if configs.encoder_output_dim > 2:
        subtitle = ("Approximated decision boundary based on the clusters in this 2D space. \n"
                    "Actual predictions might not match the background color.")
    else:
        subtitle = ("Exact decision boundary from the trained Classifier.\n"
                    "Actual predictions match the background color.")
    plt.title(f"Visualization with Classifier Decision Boundary \n{subtitle}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
