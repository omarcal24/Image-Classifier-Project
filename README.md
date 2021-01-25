# Image Classifier Project as part of the Udacity Into to Machine Learning with PyTorch Project.

I developed an application which trains a neural network using a dataset about flowers and their categories.

The main tasks of the project were:
1. Import libraries
2. Define transforms for the datasets
3. Load datasets and transform them into dataloaders
4. Load a pre-trained model (which in this case was a VGG11)
5. Freeze parameters
6. Build and train a classifier for the model
7. Set criterion NLLLoss (Because I used a LogSoftmax activation function for the output layer in my classifier)
8. Train the model and test it with a validation dataset
9. Test accuracy with the training dataset (84%)
10. Save and load the checkpoint
11. Process the image
12. Predict the category of the flower
13. Use all these previous steps I coded on a Jupyter Notebook on a command line application
