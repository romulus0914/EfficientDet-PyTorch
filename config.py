efficientdet_model_params = {
    'efficientdet-d0':
        dict(
            backbone_type='efficientnet-b0',
            image_size=512,
            fpn_num_channels=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
            weight_method='fastattn'
        ),
    'efficientdet-d1':
        dict(
            backbone_type='efficientnet-b1',
            image_size=640,
            fpn_num_channels=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
            weight_method='fastattn'
        ),
    'efficientdet-d2':
        dict(
            backbone_type='efficientnet-b2',
            image_size=768,
            fpn_num_channels=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
            weight_method='fastattn'
        ),
    'efficientdet-d3':
        dict(
            backbone_type='efficientnet-b3',
            image_size=896,
            fpn_num_channels=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
            weight_method='fastattn'
        ),
    'efficientdet-d4':
        dict(
            backbone_type='efficientnet-b4',
            image_size=1024,
            fpn_num_channels=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            weight_method='fastattn'
        ),
    'efficientdet-d5':
        dict(
            backbone_type='efficientnet-b5',
            image_size=1280,
            fpn_num_channels=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            weight_method='fastattn'
        ),
    'efficientdet-d6':
        dict(
            backbone_type='efficientnet-b6',
            image_size=1280,
            fpn_num_channels=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            weight_method='sum'
        )
}

efficientnet_model_params = {
    # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5)
}
