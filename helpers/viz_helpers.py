import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import configs
import model
from helpers.data_helpers import get_all_embeddings_or_data, restart_label_index_from_zero, create_tool_idx_list

# manually order objects by similarity
SIM_OBJECTS_LIST = ['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                    'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                    'wooden-button', 'plastic-bead', 'glass-bead', 'metal-nut-bolt']
SORTED_DATA_OBJ_LIST = sorted(SIM_OBJECTS_LIST)  # for input data from data files, the labels were created in this order


def _map_objects_to_colors(obj_list, labels):
    labels = np.squeeze(labels)
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


def _get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx):
    cbar_labels = np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)]
    reversed_mp = {sim_idx: label for label, sim_idx in label_to_subset_idx.items()}
    original_labels = [reversed_mp[sim_idx] for sim_idx in sorted(subset_obj_idx)]
    tick_labels = [f"{cbar_label}\n(label {orig_label})" for cbar_label, orig_label in
                   zip(cbar_labels, original_labels)]
    return tick_labels


def viz_data(trans_cls, encoder: model.encoder or None, data_dim=None, viz_l2_norm=configs.viz_share_space_l2_norm):
    assert (encoder or data_dim) is not None
    data_groups, label_groups, _ = get_all_embeddings_or_data(trans_cls=trans_cls, encoder=encoder, data_dim=data_dim)
    # all_emb order: source_old, source_new, assist_old, assist_new, target_old, target_new
    source_data_train, source_label_train = data_groups[0], label_groups[0]
    source_data_test, source_label_test = data_groups[1], label_groups[1]
    assist_data_train, assist_label_train = data_groups[2], label_groups[2]
    assist_data_test, assist_label_test = data_groups[3], label_groups[3]
    target_data_train, target_label_train = data_groups[4], label_groups[4]
    target_data_test, target_label_test = data_groups[5], label_groups[5]

    # viz ALL data on 2D space
    len_list = [len(l_group) for l_group in label_groups]
    tool_labels = create_tool_idx_list(source_label_len=len_list[0] + len_list[1],
                                       assist_label_train_len=len_list[2], assist_label_test_len=len_list[3],
                                       target_label_train_len=len_list[4], target_label_test_len=len_list[5])
    data_descpt = "All Input Data" if encoder is None else f"All Embedded Data-{configs.loss_func} Loss"
    _viz_embeddings(embeds=np.vstack(data_groups), labels=np.vstack(label_groups), viz_l2_norm=viz_l2_norm,
                    tool_labels=tool_labels, obj_list=configs.all_object_list, save_fig=True, title=f"{data_descpt}",
                    subtitle=f"source tool: {configs.source_tool_list}, target tool: "
                             f"{configs.target_tool_list} \n assisted tool: {configs.assist_tool_list}")
    # Viz train data on 2D space
    data = np.vstack([source_data_train, source_data_test, assist_data_train, target_data_train])
    labels = np.vstack([source_label_train, source_label_test, assist_label_train, target_label_train])
    tool_labels = create_tool_idx_list(source_label_len=len(source_label_train) + len(source_label_test),
                                       assist_label_train_len=len(assist_label_train),
                                       target_label_train_len=len(target_label_train))
    data_descpt = "Encoder Train Set Input Data" if encoder is None else f"Encoder Train Set Embedded Data-{configs.loss_func} Loos"
    _viz_embeddings(embeds=data, labels=labels, viz_l2_norm=viz_l2_norm, tool_labels=tool_labels,
                    obj_list=configs.all_object_list, save_fig=True, title=f"{data_descpt}",
                    subtitle=f"source tool: {configs.source_tool_list}, target tool: {configs.target_tool_list} \n"
                             f"assisted tool: {configs.assist_tool_list}")

    # viz test data on shared latent space
    data = np.vstack([source_data_test, assist_data_test, target_data_test])
    labels = np.squeeze(np.vstack([source_label_test, assist_label_test, target_label_test]))
    labels = restart_label_index_from_zero(labels)  # labels should start from 0 to match the new object list
    tool_labels = create_tool_idx_list(source_label_len=len(source_label_test),
                                       assist_label_test_len=len(assist_label_test),
                                       target_label_test_len=len(target_label_test))
    data_descpt = "Encoder Test Set Input Data" if encoder is None else f"Encoder Test Set Embedded Data-{configs.loss_func} Loos"
    _viz_embeddings(viz_l2_norm=viz_l2_norm, embeds=data, labels=labels, tool_labels=tool_labels,
                    obj_list=configs.all_object_list, save_fig=True, title=f"{data_descpt}",
                    subtitle=f"source tool: {configs.source_tool_list}, target tool: {configs.target_tool_list} \n"
                             f"assisted tool: {configs.assist_tool_list}")


def _viz_embeddings(embeds: np.ndarray, labels: np.ndarray, tool_labels: list, loss_func: str = configs.loss_func,
                    obj_list: list = configs.old_object_list + configs.new_object_list, viz_l2_norm=True,
                    save_fig: bool = True, title='', subtitle='', show_curr_label=False):
    """
    visualize embeddings in 2D space and use object as color and tool as marker.
    :param embeds: embedding by trial/sample:  [n_sample, emb_len]. The section of rows has to follow this order:
        ['Source Tool(old and/or new obj)', 'Assist Tool(old)', 'Assist Tool(new)', 'Target Tool(old)', 'Target Tool(new)']
    :param tool_labels: tool index for embeddings
    :param labels: [n_sample, 1 or 0], will be squeezed
    :param obj_list: list of unique object names, order aligns with labeling: e.g., ["wheat", ...] -> label 0 is "wheat"
    :param loss_func: name for the loss function, e.g. "TL"
    :param viz_l2_norm: whether visualize l2 normed embeddings, if True, embeddings will be on a unit circle
    :param save_fig: whether save this figure
    :param title: content for the first line of title
    :param subtitle: second line of title: "... Visualization of Embeddings in {save_name}\n {subtitle}"
    :param show_curr_label: show the actual label (e.g., label for classifier) for the object on color bar.
        the color bar is fixed based on SIM_OBJECTS_LIST, so we need to re-align labels with the colorbar tick labels

    :return:
    """
    logging.debug(f"➡️ viz_shared_latent_space..")
    assert (np.max(labels) == len(obj_list) and np.min(labels) == 0) or len(np.unique(obj_list)) == len(obj_list)
    labels = np.squeeze(labels)
    tool_labels = np.array(tool_labels)

    logging.debug(f"embeds shape: {embeds.shape}")
    logging.debug(f"labels: \n      {labels}")

    # make sure Dimensionality is 2D
    if embeds.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
        embeds_2d = tsne.fit_transform(embeds)
    else:
        embeds_2d = embeds

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 12
    tool_order_list = ['Source Tool(All)', 'Assist Tool(Train)', 'Assist Tool(Test)',
                       'Target Tool(Train)', 'Target Tool(Test)']
    markers = ['o', 'X', 'X', 's', 's']  # Markers for tools. Circle for source, X for assist, square for target test
    # # Create tool labels: 0 for source, 1, 2 for assist, 3,4 for target test
    # tool_labels = np.array([0] * len_dict['source'] +
    #                        [1] * len_dict['assist_train'] + [2] * len_dict['assist_test'] +
    #                        [3] * len_dict['target_train'] + [4] * len_dict['target_test'])

    # take the subset of all colors (from SIM_OBJECTS_LIST) for obj_list
    mapped_labels, cmap, norm, subset_obj_idx, label_to_subset_idx = _map_objects_to_colors(obj_list, labels)

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
        shrink = 1  # if on unit sphere, change tools' radius, so they don't block each other
        edge_color = None  # add edge to test object marker
        if tool_label in [1, 2]:  # assist embedding
            if configs.l2_norm and viz_l2_norm:
                shrink = 1.05  # outside source embedding circle
            if tool_label == 2:  # test object
                edge_color = 'gray'
        elif tool_label in [3, 4]:  # target embedding
            if configs.l2_norm and viz_l2_norm:
                shrink = 0.95  # inside source embedding circle
            if tool_label == 4:  # test object
                edge_color = 'black'
        plt.scatter(
            embeds_2d[mask, 0] * shrink, embeds_2d[mask, 1] * shrink,
            c=mapped_labels[mask],
            cmap=cmap,
            norm=norm,
            marker=marker,
            edgecolor=edge_color,
            s=80,
            alpha=1.0,
            label=f"{tool_order_list[tool_label]}"
        )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(obj_list)) + 0.5)  # ticks at the center of each color
    cbar.ax.tick_params(which='minor', length=0)  # Hide minor ticks on each color's boundary
    cbar.set_label("Objects", rotation=270, labelpad=20)
    # show labels for color bar
    if show_curr_label:  # map color bar label to input labels
        cbar.ax.set_yticklabels(_get_curr_labels_on_cbar(subset_obj_idx, label_to_subset_idx))
    else:  # colors are sorted by sim object, so should the object names
        cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])

    # plt.legend(title="Tool Type", loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.legend()
    save_name = title if title else f"Shared Space-{loss_func} loss"
    viz_discpt = "T-SNE Reduced 2D" if embeds.shape[1] > 2 else ""
    plt.title(f"{viz_discpt}Visualization of Embeddings in {save_name}\n {subtitle}")
    plt.xlabel(f"{viz_discpt}Dimension 1")
    plt.ylabel(f"{viz_discpt}Dimension 2")
    plt.grid(True)
    # plt.tight_layout()

    if save_fig:
        plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
    plt.show()
    plt.close()


def viz_test_objects_embedding(transfer_class, Encoder, Classifier, test_accuracy, pred_label_target):
    # if embeddings need feature reduction to 2D, show the actual predictions for reference
    if configs.encoder_output_dim > 2:
        pass
        # *_, pred_label_source = transfer_class.eval(Encoder=Encoder, Classifier=Classifier,  # evaluate source tool
        #                                             tool_list=configs.source_tool_list, return_pred=True)
        # all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
        #     Encoder=Encoder, new_obj_only=True)
        # labels = np.concatenate([pred_label_source.cpu().detach().numpy(),
        #                          pred_label_target.cpu().detach().numpy()], axis=0)
        # viz_embeddings(obj_list=configs.new_object_list, embeds=all_embeds, labels=labels,
        #                len_dic=[source_len, target_len, target_test_len], show_curr_label=True,
        #                subtitle=f"Test Predictions. Accuracy: {test_accuracy*100:.1}%, Target {configs.target_tool_list}"
        #                         f" \n Source: {configs.source_tool_list}")

    # visualize the data in 2 D space. The original labels are preserved because there will be colored background for predicted labels
    # If the embedding is 2D, use the trained Classifier for decision boundaries, the background color will match the actual predictions
    # if > 2, the boundaries are from a logistic regression clf trained on t-sne reduced 2D space,
    #   the background does not reflect actual predictions, it approximates what a linear classifier does on full-sized embeddings.
    _viz_classifier_boundary_on_2d_embeddings(transfer_class, Encoder, Classifier, accuracy=test_accuracy)


def _viz_classifier_boundary_on_2d_embeddings(transfer_class, Encoder, Classifier, accuracy=None):
    vis_l2_norm = configs.l2_norm
    object_list = configs.new_object_list
    # Step 1: Generate embedded data
    all_emb, all_labels, _ = get_all_embeddings_or_data(trans_cls=transfer_class, encoder=Encoder, old_object_list=[])
    source_emb, source_y = all_emb[1], all_labels[1]
    assist_emb, assist_y = all_emb[3], all_labels[3]
    target_emb, target_y = all_emb[5], all_labels[5]
    all_emb = np.vstack([source_emb, assist_emb, target_emb])

    # Step 2: make sure the dimension is 2
    if source_emb.shape[1] > 2:  # Apply T-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
        all_embeddings_2d = tsne.fit_transform(all_emb)
        embeddings_2d = all_embeddings_2d[:len(source_emb)]
        assist_embeddings_2d = all_embeddings_2d[len(source_emb): len(target_emb)]
        new_embeddings_2d = all_embeddings_2d[-len(target_emb):]
    elif source_emb.shape[1] == 2:
        embeddings_2d = source_emb
        assist_embeddings_2d = assist_emb
        new_embeddings_2d = target_emb
        all_embeddings_2d = all_emb
    else:
        raise Exception(f"embedding shape is not correct: {all_emb.shape}")

    # Step 3: Create a grid for decision boundary on 2d space
    if configs.l2_norm:
        buffer = 0.1
        x_min, x_max = -1 - buffer, 1 + buffer
        y_min, y_max = -1 - buffer, 1 + buffer
    else:
        buffer = 1
        x_min, x_max = all_embeddings_2d[:, 0].min() - buffer, all_embeddings_2d[:, 0].max() + buffer
        y_min, y_max = all_embeddings_2d[:, 1].min() - buffer, all_embeddings_2d[:, 1].max() + buffer
    n_points = 500
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # (n_points*n_points, 2)

    # Step 4: Train a classifier in the 2D T-SNE space
    if all_emb.shape[1] == 2:  # use the trained classifier directly because input shape matches
        grid_proba = Classifier(torch.tensor(grid_points, dtype=torch.float32, device=configs.device))
        grid_proba = grid_proba.cpu().detach().numpy()  # (n_points*n_points, C)
    else:  # use LogisticRegression to classify data on the t-sne transformed space (approximate the decision boundaries of the Classifier)
        secondary_classifier = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
        secondary_classifier.fit(embeddings_2d, source_y)
        grid_proba = secondary_classifier.predict_proba(grid_points)  # Shape: [n_grid_points, C]

    grid_classes = np.argmax(grid_proba, axis=1)  # (n_points*n_points, )

    # Step 5: Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 12

    levels = np.arange(-1, len(object_list))  # strangely it needs the lowest level as -1 to show all colors!!!
    grid_classes, cmap, norm, subset_obj_idx, label_to_subset_idx = _map_objects_to_colors(object_list, grid_classes)
    plt.contourf(xx, yy, grid_classes.reshape(xx.shape), cmap=cmap, levels=levels,
                 alpha=0.3)  # contourf flips the array upside down

    # Step 6: Overlay the original and test embeddings
    color_label_source, *_ = _map_objects_to_colors(object_list, source_y)
    color_label_target, *_ = _map_objects_to_colors(object_list, target_y)
    color_label_assist, *_ = _map_objects_to_colors(object_list, assist_y)
    scatter_original = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=color_label_source, cmap=cmap, norm=norm, s=50, alpha=1.0, label="Source Tool"
    )
    # if on unit sphere, shrink the target tool embeddings' sphere, so they don't block the source embeddings
    shrink = 0.95 if configs.l2_norm else 1
    plt.scatter(
        new_embeddings_2d[:, 0] * shrink, new_embeddings_2d[:, 1] * shrink,
        c=color_label_target, cmap=cmap, norm=norm, edgecolor='k', s=80, alpha=1.0, marker='s', label="Target Tool"
    )
    if "assist" in configs.clf_exp_name:
        shrink = 1.05 if configs.l2_norm else 1
        plt.scatter(
            assist_embeddings_2d[:, 0] * shrink, assist_embeddings_2d[:, 1] * shrink,
            c=color_label_assist, cmap=cmap, norm=norm, edgecolor='gray', s=80, alpha=1.0, marker='X',
            label="Assist Tool"
        )

    # Add a color bar and legend
    cbar = plt.colorbar(scatter_original, ticks=np.arange(len(object_list)) + 0.5)  # ticks at the center of each color
    cbar.ax.tick_params(which='minor', length=0)  # Hide minor ticks on each color's boundary
    cbar.ax.set_yticklabels(np.array(SIM_OBJECTS_LIST)[sorted(subset_obj_idx)])
    cbar.set_label("Classes", rotation=270, labelpad=20)

    if configs.encoder_output_dim > 2:
        subtitle = ("Approximated decision boundary based on the clusters in this 2D space. \n"
                    f"Actual predictions might not match the background color. Accuracy: {accuracy * 100:.1f}%")
    else:
        subtitle = ("Exact decision boundary from the trained Classifier.\n"
                    f"Actual predictions match the background color. Accuracy: {accuracy * 100:.1f}%")
    plt.title(f"Visualization with Classifier Decision Boundary on Test Set - {configs.loss_func} \n{subtitle}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    if vis_l2_norm:
        plt.xlim(-1 - 0.1, 1 + 0.1)
        plt.ylim(-1 - 0.1, 1 + 0.1)
    plt.show()

