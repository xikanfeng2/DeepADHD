import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras import backend as K
import sys
from deepexplain.tensorflow import DeepExplain

# load trained model
model = keras.models.load_model('cnn_model.h5')
model.summary()

# read snp data
snps = pd.read_csv('input.csv')

# process testing dataset
test_index = list(pd.read_csv('test_index.csv', header=None)[0])
test = snps.iloc[test_index]
test_data = test.iloc[:, 6:].values
test_data = test_data.astype(np.float16)
test_labels = test.iloc[:, 5].values
test_labels = test_labels.astype(np.int32)
test_labels = test_labels - 1

loss, acc = model.evaluate(test_data, test_labels)

print("Restored model, accuracy: {:5.2f}%".format(100*acc))


predicted_labels = model.predict_classes(test_data)
result = []
for i in predicted_labels:
    result.append(i[0])
result = np.asarray(result)

diff = result - test_labels
diff_indexes = np.where(diff != 0)

remained_data = np.delete(test_data, diff_indexes, axis=0)
remained_labels = np.delete(test_labels, diff_indexes)

with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = model.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
    target_tensor = fModel(input_tensor)

    xs = remained_data
    ys = remained_labels.reshape(len(remained_labels), 1)
    
    # using saliency method to conduct saliency analysis
    attributions_sal = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys) 

    # store the saliency analysis result
    pd.DataFrame(attributions_sal).to_csv('saliency-analysis.csv', index=False)
