
# not used in the final report

def one_crowd(train_images, train_labels, test_images, test_labels, loop_num=10, args=args):
    # Get the images
    best_models = []
    model_accs = []
    model_confs = []
    many_labels = []

    # Run the CNN denoise training loop
    for seed in range(loop_num):
        results, _, predicted_labels, _, _, _ = one_cnn_run(args, seed, train_images, train_labels, test_images, test_labels)
        best_models.append(results['best_model'])
        model_accs.append(results['stats']['test_acc'])
        model_confs.append(results['stats']['test_conf'])
        labels = []
        for k in range(len(predicted_labels)):
            labels.extend(predicted_labels[k].tolist())
        many_labels.append(np.array(labels))
    return best_models, model_accs, model_confs, many_labels

def crowd_stats(loop_num=10,crowd_num=10,binary=True):
    # use many image random states to generate many crowds and collect their stats
    crowd_accs = []
    crowd_confs = []
    crowd_acc_means = []
    crowd_conf_means = []
    for i in range(crowd_num):
        train_images, train_labels, test_images, test_labels = get_images(i+42, binary=binary)
        _, model_accs, model_confs, many_labels = one_crowd(train_images, train_labels, test_images, test_labels, loop_num)
        crowd_accs.append(model_accs)
        crowd_confs.append(model_confs)
        crowd_acc_means.append(np.mean(model_accs))
        crowd_conf_means.append(np.mean(model_confs))
        print(f'crowd {i+1} done')
    return crowd_accs, crowd_confs, crowd_acc_means, crowd_conf_means

