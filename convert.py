import coremltools
import sympy

# input_tensor_shapes = {"input:0": [1, 150, 150, 3]}
# image_input_name = ['input:0']
# coreml_model_file = 'models/hotdogmodel.mlmodel'
# output_tensor_names = ['Softtmax:0']
class_labels = ['hot_dog', 'not_hot_dog']

coremltools.converters.tensorflow.convert('models/save_at_30.h5',
                                        )
coreml_model = coremltools.converters.tensorflow.convert('models/save_at_30.h5',
                                                         image_input_names='conv2d_input',
                                                         class_labels=class_labels,
                                                         image_scale=1.)


for desc in coreml_model.input_description:
    print("input desc: {}".format(desc))
for desc in coreml_model.output_description:
    print("output desc: {}".format(desc))

coreml_model.author = 'Enrique Aviña'
coreml_model.license = 'BSD'
coreml_model.short_description = 'Model to classify hot dogs'
coreml_model.input_description['conv2d_input'] = '1x150x150x3 (1 batch sample of an RGB 150x150 image)'
coreml_model.output_description['classLabel'] = 'Whether image is hot dog or not'
coreml_model.save('models/nicehotdog.mlmodel')
