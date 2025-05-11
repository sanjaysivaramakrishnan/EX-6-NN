<H3>ENTER YOUR NAME: SANJAY SIVARAMAKRISHNAN M</H3>
<H3>ENTER YOUR REGISTER NO: 212223240151</H3>
<H3>EX. NO.6</H3>
<H3>DATE: 11.05.2025</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>

```
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
df = pd.read_csv(r'https://raw.githubusercontent.com/Lavanyajoyce/EX-6-NN/main/heart.csv')
df.head()
X = df.drop(columns='target')
y = df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)
print('X_test : ',X_test.shape)
print('y_test : ',y_test.shape)
# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train,y_train).loss_curve_
# Make predictions on the testing set
y_pred = mlp.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test,y_pred)
confusion_mtx = confusion_matrix(y_test,y_pred)
classification_rep = classification_report(y_test,y_pred)
print('Accuracy: ',accuracy)
print('Confusion Matrix : ', confusion_mtx)
print('Classification Report : ',classification_rep)
# plot the error convergence
plt.plot(training_loss)
plt.title('MLP Training loss Convergence')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.show()

```

<H3>Output:</H3>

![image](https://github.com/user-attachments/assets/ec9a4cc4-7cae-47ab-b759-2c0f1a8e99c8)
![image](https://github.com/user-attachments/assets/eca3e6b0-60cb-4332-a1c7-6384d926867b)
![image](https://github.com/user-attachments/assets/4ea20656-dc48-421f-b76e-bfbf87841fc5)
![image](https://github.com/user-attachments/assets/906ab2f2-acd0-4d94-abdb-aca8614594e6)
![image](https://github.com/user-attachments/assets/a98d214e-95ee-4043-bfd4-7796006493e5)


<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
