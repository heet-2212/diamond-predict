import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("diamonds.csv")

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define a function to make predictions
def predict_price(carat, depth, table, x, y, z, cut, color, clarity):
    # Encode the categorical variables for input data
    input_data = {
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z],
        'cut_Ideal': [1 if cut == 'Ideal' else 0],
        'cut_Premium': [1 if cut == 'Premium' else 0],
        'cut_Good': [1 if cut == 'Good' else 0],
        'color_E': [1 if color == 'E' else 0],
        'clarity_SI1': [1 if clarity == 'SI1' else 0],
        'clarity_VS1': [1 if clarity == 'VS1' else 0]
        # Add any other categorical features that were present during training
    }

    input_df = pd.DataFrame(input_data)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    return model.predict(input_df)[0]


# Streamlit app with sidebar
st.sidebar.title('Prediction System')
page = st.sidebar.radio('Select a page:', ['Home'])

if page == 'Home':

    # Streamlit app
    st.title('Diamond Price Predictor')

    # Input features
    carat = st.number_input('Carat', min_value=0.0, step=0.01)
    depth = st.number_input('Depth', min_value=0.0, step=0.1)
    table = st.number_input('Table', min_value=0.0, step=0.1)
    x = st.number_input('x', min_value=0.0, step=0.01)
    y = st.number_input('y', min_value=0.0, step=0.01)
    z = st.number_input('z', min_value=0.0, step=0.01)

    cut = st.selectbox('Cut', ['Ideal', 'Premium', 'Good'])
    color = st.selectbox('Color', ['E', 'G', 'F', 'H', 'D'])
    clarity = st.selectbox('Clarity', ['SI1', 'VS1', 'SI2', 'VS2', 'VVS1', 'VVS2', 'IF'])

    if st.button('Predict'):
        predicted_price = predict_price(carat, depth, table, x, y, z, cut, color, clarity)
        with st.container():
            st.markdown(f"<h2 style='color: blue;'>Predicted price: ${predicted_price:.2f}</h2>", unsafe_allow_html=True)

    # Define the content of your footer
    footer_content = """
    <footer style="text-align: center; margin-top: 20px;">
        <hr>
        <p> Diamonmds Price Prediction @ copywrite </p>
        <p> Developed By Heet Shah || Contact - heetshah123@gmail.com </p>
    </footer>
    """

    # Display the footer using markdown
    st.markdown(footer_content, unsafe_allow_html=True)
