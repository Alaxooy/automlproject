# automlproject
 Automate machine learning .

Auto ML COEX
Auto ML COEX is a Streamlit web application that helps you perform Automated Machine Learning (AutoML) on your dataset. It allows you to upload a CSV dataset, explore the data, run AutoML to find the best model, and visualize the model performance metrics.

Features
Upload Dataset: Upload your dataset in CSV format.
Data Exploration: Explore the uploaded dataset using D-Tale for data profiling.
Automated Machine Learning: Run AutoML using PyCaret to find the best model for your data.
Model Performance Visualization: Visualize the performance of different models in terms of Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/auto-ml-coex.git
Navigate to the project directory:

arduino
Copy code
cd auto-ml-coex
Install the required dependencies:

Copy code
pip install -r requirements.txt
Usage
Run the Streamlit app:

arduino
Copy code
streamlit run app.py
Upload your dataset and follow the instructions on the web app to explore data and run AutoML.

Technologies Used
Streamlit: Web app framework for building the user interface.
Pandas: Data manipulation and analysis library.
PyCaret: Automated machine learning library for model training and selection.
D-Tale: Interactive data visualization tool for data profiling.
Matplotlib and Seaborn: Plotting libraries for data visualization.
Folder Structure
bash
Copy code
auto-ml-coex/
│
├── app.py              # Streamlit application code
├── requirements.txt    # Python dependencies
├── README.md           # Project README
└── data/               # Directory to store dataset (optional)
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
