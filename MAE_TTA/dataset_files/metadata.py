"""
    From the original dataset generation:
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
"""

corruption_names_train = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                          'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
                          ]

corruption_names_val = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
                        ]