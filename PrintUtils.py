
import numpy as np
import matplotlib.pyplot as plt

class PrintUtils   :


  def plot_dataPoint(self,i,class_names, predictions_array, true_label, data):
    predictions_array, true_label, data = predictions_array, true_label[i], data[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    #plt.imshow(data, cmap=plt.cm.binary)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

  def plot_value_array(self,i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    4
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
  def plotMultiPredictions(self,predictions,test_labels,test_data):
    num_rows = 3
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        self.plot_dataPoint(i,range(10), predictions[i], test_labels, test_data)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        self.plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
    plt.show()
