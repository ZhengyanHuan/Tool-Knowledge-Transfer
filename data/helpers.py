import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE
import torch

import model
import configs
from transfer_class import Tool_Knowledge_transfer_class


def make_emb_data(X_array, Y_array, behavior_list, tool_list, obj_list, data):
    X_array = [X_array] if len(X_array) != 0 else []
    Y_array = [Y_array] if len(Y_array) != 0 else []
    meta_data = {b: {t: {} for t in tool_list} for b in behavior_list}
    for b_idx, b in enumerate(behavior_list):
        for t_idx, t in enumerate(tool_list):
            for o_idx, o in enumerate(obj_list):
                X_array.append(data[b][t]['audio'][o]['X'])
                Y_array.append(data[b][t]['audio'][o]['Y'])
                meta_data[b][t][o] = len(data[b][t]['audio'][o]['Y'])
    print(f"data input meta_data: {meta_data}")
    return np.vstack(X_array), np.vstack(Y_array)


def viz_input_data(data, loss_func_name: str, behavior_list, source_tool_list, target_tool_list, old_object_list,
                   new_object_list, shared_only: bool, test_only: bool, plot_title="Original Data Space"):
    assert ~(shared_only and test_only)
    if shared_only:
        all_obj_list = old_object_list
    elif test_only:
        all_obj_list = new_object_list
    else:
        all_obj_list = old_object_list + new_object_list

    X_array, Y_array = make_emb_data([], [], behavior_list, source_tool_list, all_obj_list, data=data)
    len_source = len(source_tool_list) * len(all_obj_list) * 10
    len_target_shared = len(old_object_list) * 10
    len_target_test = len(new_object_list) * 10
    if not test_only:
        X_array, Y_array = make_emb_data(X_array, Y_array, behavior_list, target_tool_list, old_object_list, data=data)
        if shared_only:
            len_target_test = 0
    if not shared_only:
        X_array, Y_array = make_emb_data(X_array, Y_array, behavior_list, target_tool_list, new_object_list, data=data)
        if test_only:
            len_target_shared = 0
    viz_shared_latent_space(loss_func=loss_func_name, all_obj_list=all_obj_list,
                            all_embeds_1D=X_array, all_labels=Y_array, save_fig=False,
                            len_list=[len_source, len_target_shared, len_target_test], title=plot_title)


def viz_embeddings(transfer_class: Tool_Knowledge_transfer_class, loss_func, viz_objects: str, input_dim,
                   source_tool_list, target_tool_list, modality_list, trail_list, behavior_list,
                   old_object_list, new_object_list, encoder_state_dict_loc: str = './saved_model/encoder/'):
    encoder_pt_name = f"myencoder_{loss_func}.pt"
    Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                            hidden_size=configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(
        torch.load(encoder_state_dict_loc + encoder_pt_name, map_location=torch.device(configs.device)))
    if viz_objects == "shared":
        all_obj_list = old_object_list + new_object_list
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder, new_obj_only=False, train_obj_only=True, behavior_list=behavior_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            modality_list=modality_list, old_object_list=old_object_list, new_object_list=new_object_list,
            trail_list=trail_list)
        viz_shared_latent_space(
            loss_func=loss_func, all_obj_list=all_obj_list, all_embeds_1D=all_embeds, all_labels=all_labels,
            len_list=[source_len, target_len, target_test_len], save_fig=False)
    if viz_objects == "test":
        all_obj_list = [] + new_object_list
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder, new_obj_only=True, behavior_list=behavior_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            modality_list=modality_list, old_object_list=[], new_object_list=new_object_list, trail_list=trail_list)
        viz_shared_latent_space(
            loss_func=loss_func, all_obj_list=all_obj_list, all_embeds_1D=all_embeds, all_labels=all_labels,
            len_list=[source_len, target_len, target_test_len], save_fig=False)
    if viz_objects == "all":
        all_obj_list = old_object_list + new_object_list
        all_embeds, all_labels, source_len, target_len, target_test_len = transfer_class.encode_all_data(
            Encoder, behavior_list, source_tool_list, target_tool_list, modality_list,
            old_object_list, new_object_list, trail_list)
        viz_shared_latent_space(
            loss_func=loss_func, all_obj_list=all_obj_list, all_embeds_1D=all_embeds, all_labels=all_labels,
            len_list=[source_len, target_len, target_test_len])


def viz_shared_latent_space(loss_func: str, all_obj_list: list, all_embeds_1D: np.ndarray,
                            all_labels: np.ndarray, len_list: list, save_fig: bool = True, title='') -> None:
    """

    :param loss_func: name for the loss function
    :param all_obj_list: set of all object names, in the order of old + new
    :param all_embeds_1D:  array of all 1D embeddings from source (all and/or new obj), target (old obj), and target test (new obj),
                        shape=(sum(len_list), len_1D_emb)
    :param all_labels:
    :param len_list: length of data from each tool&(object set) combo:
                        [n_emb_source_tool_source_objects, n_emb_target_tool_old_obj, n_emb_target_tool_new_obj]:
                        (n_source_tools * n_source_objects * n_trials, n_old_objects * n_trials, n_new_objects * n*n_trials)
    :param save_fig: save the fig or not
    :param title: customized title section, following "T-SNE Visualization of Embeddings in "
    :return: None
    """
    print(f"➡️ viz_shared_latent_space..")
    source_len, target_len, target_test_len = len_list
    all_labels = np.squeeze(all_labels, axis=None)
    # Create tool labels: 0 for source, 1 for target, 2 for target test
    tool_labels = np.array([0] * source_len + [1] * target_len + [2] * target_test_len)

    print(f"all_embeds_1D: {all_embeds_1D.shape}")
    print(f"all_labels: {all_labels}")
    print(f"all_obj_list: {all_obj_list}")
    print(f"source_len, target_len, target_test_len: {source_len, target_len, target_test_len}")

    # Step 1: Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate="auto")
    embeds_2d = tsne.fit_transform(all_embeds_1D)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    markers = ['o', '^', 'x']  # Markers for tools. Circle for source, triangle for target, x for target test
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_obj_list)))
    cmap = ListedColormap(colors)  # Discrete colormap
    bounds = list(range(len(all_obj_list) + 1))  # Boundaries for discrete colors
    norm = BoundaryNorm(bounds, cmap.N)  # Normalize to discrete boundaries

    scatter = plt.scatter(
        embeds_2d[:, 0], embeds_2d[:, 1],
        c=all_labels,
        cmap=cmap,  # Use discrete colormap
        norm=norm,  # Apply discrete normalization
        s=0  # Use invisible points for the color bar reference
    )

    for tool_label, marker in enumerate(markers):
        mask = tool_labels == tool_label
        plt.scatter(
            embeds_2d[mask, 0], embeds_2d[mask, 1],
            c=all_labels[mask],
            # without discrete normalization, cmap will align labels differently when they don't have the same range (i.e. labels [0, 3] range is not 15)
            cmap=cmap,  # Use discrete colormap
            norm=norm,  # Apply discrete normalization
            marker=marker,
            s=50,
            alpha=0.7,
            label=f"{['Source Tool(All)', 'Target Tool(Train)', 'Target Tool(Test)'][tool_label]}"
        )

    # Step 3: Add legend for tools (shapes)
    plt.legend(title="Tool Type")

    # Step 4: Add color bar for objects
    cbar = plt.colorbar(scatter, ticks=range(len(all_obj_list)))
    cbar.ax.set_yticklabels(all_obj_list)
    cbar.set_label("Objects", rotation=270, labelpad=20)

    # Plot details
    if title:
        save_name = title
    else:
        save_name = f"shared_space-{loss_func} loss"
    plt.title(f"T-SNE Visualization of Embeddings in {save_name}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    if save_fig:
        plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
