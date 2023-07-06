import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('DDos.csv')

X = df.drop(' Label', axis=1)
variances = X.var()
threshold = 0.1
selected_features = variances[variances > threshold].index.tolist()
selected_columns = selected_features

df = df[selected_columns + [' Label']]
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df[' Label'])
label_names = label_encoder.classes_


X = df.drop([' Label', 'target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_tensor = torch.Tensor(X_train_scaled)
X_test_tensor = torch.Tensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(label_names))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Plotting the loss
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    predicted_labels = label_names[predicted]
    y_test_labels = label_names[y_test_tensor]
    print(classification_report(y_test_labels, predicted_labels, zero_division=1))
    from sklearn.metrics import accuracy_score, confusion_matrix

    accuracy = accuracy_score(y_test_labels, predicted_labels)
    confusion = confusion_matrix(y_test_labels, predicted_labels)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion)

report_df = pd.DataFrame.from_dict(classification_report(y_test_labels, predicted_labels, output_dict=True))
report_df = report_df.iloc[:-1, :].T

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap='Blues', ax=ax)
ax.set_title('Classification Report')
plt.show()