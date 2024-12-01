# Labels are not fixed in this project.
## Overview
  - we don't use the original labels from the dataset.
  - all labels are created ad hoc
    - label index starts from 0 to number of given objects
      - labels for classifiers should have the range by number of objects when we only use a subset of objects.
    - labels changes when task changes:
        - encoder label 0-15 when there are 15 objects
        - classifier labels 0-5 given 5 test objects, these 0-5 are different from those 0-5 for the encoder

## Labeling Process
- original labels from the dataset are indexed by **sorted** 15 objects
  - ``SORTED_DATA_OBJ_LIST = ['cane-sugar', 'chia-seed', 'chickpea', 'detergent', 'empty', 'glass-bead', 'kidney-bean', 'metal-nut-bolt', 'plastic-bead', 'salt', 'split-green-pea', 'styrofoam-bead', 'water', 'wheat', 'wooden-button']``
  - 'cane-sugar's label is 0, 'wooden-button' is 14.
- current labels are created by the order of objects in `all_object_list`, where `all_object_list = old_object_list + new_object_list`
  - `old_object_list` has objects shared by source and target tools, and it's only used for encoder training. `new_object_list` has objects for testing, and during training, only source tool has access to these objects.
  - there is no overlap between `old_object_list` and `new_object_list`
  - encoding does not need consistent labeling because labels are used to group/differentiate samples
    - e.g., during training, train set objects 0-15 are from source tools, and the subset objects 0-10 are from target tool; during testing, we no longer need the labels, if labeling is needed for some reason, the test set of objects can start from 0 again.
  - classifiers does not have to follow the labels from encoder
    - classifiers only takes the test set objects from source and target tools, so the labels start from 0.
- for visualization, object and color mapping is fixed
    - for better understanding and easier comparison, similar (by sound) objects have similar colors
    - convert current labels to color bar labels using a list where similar objects are closer: 
      * `SIM_OBJECTS_LIST = ['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                        'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                        'wooden-button', 'glass-bead', 'plastic-bead', 'metal-nut-bolt']`
      * 'empty' matches the first color, 'metal-nut-bolt' matches the last color of a fixed color set
  