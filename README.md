# FastApi_ModelDeployment

## Simple Transaction Prediction Model

**Version**: 0.1.0

**Description**: This is a simple transaction prediction model using FastAPI. It takes the transaction date and amount as input and predicts whether the transaction is fraudulent or not.

### Overview

This project demonstrates how to deploy a machine learning model using FastAPI, with GitHub Actions for automated unit testing. It utilizes the FastAPI framework to create an API that accepts transaction information and returns predictions based on a trained model.

### How it Works

1. **Requirements**: Make sure you have the necessary libraries installed, including FastAPI, Pandas, and others as listed in the `requirements.txt` file.

2. **Clone the Repository**: Clone this repository to your local machine.

3. **Installation**: Run `pip install -r requirements.txt` to install the dependencies.

4. **Execution**: Start the application with `uvicorn main:app --reload`. Access the API at `http://localhost:8000`.

### API Routes

- **Root Route (/)**: Returns a welcome message.

- **Prediction Route (/predict)**: Accepts transaction date and amount as input and makes a prediction about the authenticity of the transaction.

### Project Structure

- `main.py`: The main file that sets up the FastAPI application and defines the routes.
- `model.py`: Contains the `DataModeler` class that prepares, trains, and makes predictions using the machine learning model.
- `requirements.txt`: List of project dependencies.

### Customization

You can customize this project to fit your specific machine learning model. Replace the trained model and adjust preprocessing as needed.
