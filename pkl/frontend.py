import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
def loadModel():
    return pickle.load(open('SMOTE_Voting.pkl', 'rb'))

def displayHeader():
    components.html(
        """
        <h1 style="color: white; font-size: 24px; text-align: center; font-family: sans-serif;">Predicting students' dropout and academic success</h1>
        """,
        height=50,
    )

def getUserInput():
    maritalStatus = st.slider("Marital Status", min_value=1, max_value=6, value=1)
    applicationMode = st.select_slider("Application Mode", options=[1, 2, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57])
    applicationOrder = st.slider("Application Order", min_value=1, max_value=9, value=1)
    course = st.select_slider("Course", options=[33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991])
    dayEveningAttendance = st.slider("Daytime/Evening Attendance", min_value=0, max_value=1, value=0)
    prevQual = st.select_slider("Previous Qualification", options=[1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43])
    prevQualGrade = st.text_input("Previous Qualification (Grade)")

    if prevQualGrade != '':
        prevQualGrade = float(prevQualGrade)

    admissionGrade = st.text_input("Admission Grade")

    if admissionGrade != '':
        admissionGrade = float(admissionGrade)

    displaced = st.slider("Displaced", min_value=0, max_value=1, value=0)
    debtor = st.slider("Debtor", min_value=0, max_value=1, value=0)
    tuitionUptoDate = st.slider("Tuition fees up to date", min_value=0, max_value=1, value=0)
    gender = st.slider("Gender", min_value=0, max_value=1, value=0)
    scholarshipHolder = st.slider("Scholarship Holder", min_value=0, max_value=1, value=0)
    ageAtEnrollment = st.slider("Age at Enrollment", min_value=17, max_value=70, value=18)
    curricularFirstSemCredited = st.slider("Curricular Units 1st semester (credited)", min_value=0, max_value=20, value=0)
    curricularFirstSemEnrolled = st.slider("Curricular Units 1st semester (enrolled)", min_value=0, max_value=20, value=0)
    curricularFirstSemEval = st.slider("Curricular Units 1st semester (evaluations)", min_value=0, max_value=20, value=0)
    curricularFirstSemApproved = st.slider("Curricular Units 1st semester (approved))", min_value=0, max_value=20, value=0)
    curricularSecondSemCredited = st.slider("Curricular Units 2nd semester (credited)", min_value=0, max_value=20, value=0)
    curricularSecondSemEnrolled = st.slider("Curricular Units 2nd semester (enrolled)", min_value=0, max_value=20, value=0)
    curricularSecondSemEval = st.slider("Curricular Units 2nd semester (evaluations)", min_value=0, max_value=20, value=0)
    curricularSecondSemApproved = st.slider("Curricular Units 2nd semester (approved)", min_value=0, max_value=20, value=0)
    gdp = st.text_input("GDP")

    if gdp != '':
        gdp = float(gdp)

    curricularFirstSemGrade = st.text_input("Curricular Units 1st semester (grade)")

    if curricularFirstSemGrade != '':
        curricularFirstSemGrade = float(curricularFirstSemGrade)

    curricularSecondSemGrade = st.text_input("Curricular Units 2nd semester (grade)")

    if curricularSecondSemGrade != '':
        curricularSecondSemGrade = float(curricularSecondSemGrade)

    return {
    'maritalStatus': maritalStatus,
    'applicationMode': applicationMode,
    'applicationOrder': applicationOrder,
    'course': course,
    'dayEveningAttendance': dayEveningAttendance,
    'prevQual': prevQual,
    'prevQualGrade': prevQualGrade,
    'admissionGrade': admissionGrade,
    'displaced': displaced,
    'debtor': debtor,
    'tuitionUptoDate': tuitionUptoDate,
    'gender': gender,
    'scholarshipHolder': scholarshipHolder,
    'ageAtEnrollment': ageAtEnrollment,
    'curricularFirstSemCredited': curricularFirstSemCredited,
    'curricularFirstSemEnrolled': curricularFirstSemEnrolled,
    'curricularFirstSemEval': curricularFirstSemEval,
    'curricularFirstSemApproved': curricularFirstSemApproved,
    'curricularFirstSemGrade': curricularFirstSemGrade,
    'curricularSecondSemCredited': curricularSecondSemCredited,
    'curricularSecondSemEnrolled': curricularSecondSemEnrolled,
    'curricularSecondSemEval': curricularSecondSemEval,
    'curricularSecondSemApproved': curricularSecondSemApproved,
    'curricularSecondSemGrade': curricularSecondSemGrade,
    'gdp': gdp,
}


def displayKeySidebar():
    
    st.sidebar.title("Key")

    st.sidebar.header('Marital Status')
    maritalStatusOptions = {
        '1': 'Single',
        '2': 'Married',
        '3': 'Widower',
        '4': 'Divorced',
        '5': 'Facto Union',
        '6': 'Legally Separated',
    }
    for key, value in maritalStatusOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Application Mode')
    applicationModeOptions = {
        '1': '1st phase - general contingent',
        '2': 'Ordinance No. 612/93',
        '5': '1st phase - special contingent (Azores Island)',
        '7': 'Holders of other higher courses',
        '10': 'Ordinance No. 854-B/99',
        '15': 'International student (bachelor)',
        '16': '1st phase - special contingent (Madeira Island)',
        '17': '2nd phase - general contingent',
        '18': '3rd phase - general contingent',
        '26': 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        '27': 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        '39': 'Over 23 years old',
        '42': 'Transfer',
        '43': 'Change of course',
        '44': 'Technological specialization diploma holders',
        '51': 'Change of institution/course',
        '53': 'Short cycle diploma holders',
        '57': 'Change of institution/course (International)',
    }
    for key, value in applicationModeOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Application Order')
    applicationOrderOptions = {
        '0': 'first choice',
        '1': '2nd choice',
        '2': '3rd choice',
        '3': '4th choice',
        '4': '5th choice',
        '5': '6th choice',
        '6': '7th choice',
        '7': '8th choice',
        '8': '9th choice',
        '9': 'last choice',
    }
    for key, value in applicationOrderOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Course')
    courseOptions = {
        '33': 'Biofuel Production Technologies',
        '171': 'Animation and Multimedia Design',
        '8014': 'Social Service (evening attendance)',
        '9003': 'Agronomy',
        '9070': 'Communication Design',
        '9085': 'Veterinary Nursing',
        '9119': 'Informatics Engineering',
        '9130': 'Equinculture',
        '9147': 'Management',
        '9238': 'Social Service',
        '9254': 'Tourism',
        '9500': 'Nursing',
        '9556': 'Oral Hygiene',
        '9670': 'Advertising and Marketing Management',
        '9773': 'Journalism and Communication',
        '9853': 'Basic Education',
        '9991': 'Management (evening attendance)',
    }
    for key, value in courseOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Daytime/Evening Attendance')
    dayTimeEveningOptions = {
        '0': 'Evening',
        '1': 'Daytime',
    }
    for key, value in dayTimeEveningOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Previous Qualification')
    prevQualOptions = {
        '1': 'Secondary education',
        '2': "Higher education - bachelor's degree",
        '3': "Higher education - degree",
        '4': "Higher education - master's",
        '5': "Higher education - doctorate",
        '6': "Frequency of higher education",
        '9': "12th year of schooling - not completed",
        '10': "11th year of schooling - not completed",
        '12': "Other - 11th year of schooling",
        '14': "10th year of schooling",
        '15': "10th year of schooling - not completed",
        '19': "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        '38': "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        '39': "Technological specialization course",
        '40': "Higher education - degree (1st cycle)",
        '42': "Professional higher technical course",
        '43': "Higher education - master (2nd cycle)",
    }
    for key, value in prevQualOptions.items():
        st.sidebar.text(f"{key} - {value}")


    st.sidebar.header('Previous Qualification (Grade)')
    st.sidebar.text('Grade of previous qualification (between 0 and 200)')

    st.sidebar.header('Admission Grade')
    st.sidebar.text('Admission grade (between 0 and 200)')

    st.sidebar.header('Displaced')
    displacedOptions = {
        '0': 'no',
        '1': "yes",
    }
    for key, value in displacedOptions.items():
        st.sidebar.text(f"{key} - {value}")

    st.sidebar.header('Debtor')
    debtorOptions = {
        '0': 'no',
        '1': "yes",
    }
    for key, value in debtorOptions.items():
        st.sidebar.text(f"{key} - {value}")

    st.sidebar.header('Tuition fees up to date')
    tuitionUpToDateOptions = {
        '0': 'no',
        '1': "yes",
    }
    for key, value in tuitionUpToDateOptions.items():
        st.sidebar.text(f"{key} - {value}")

    st.sidebar.header('Gender')
    genderOptions = {
        '0': 'female',
        '1': "male",
    }
    for key, value in genderOptions.items():
        st.sidebar.text(f"{key} - {value}")

    st.sidebar.header('Scholarship Holder')
    scholarshipHolderOptions = {
        '0': 'no',
        '1': "yes",
    }
    for key, value in scholarshipHolderOptions.items():
        st.sidebar.text(f"{key} - {value}")

    st.sidebar.header('Age at enrollment')
    st.sidebar.text('Age of student at enrollment')

    st.sidebar.header('Curricular units 1st semester (credited)')
    st.sidebar.text('Number of curricular units credited in the 1st semester')

    st.sidebar.header('Curricular units 1st semester (enrolled)')
    st.sidebar.text('Number of curricular units enrolled in the 1st semester')

    st.sidebar.header('Curricular units 1st semester (evaluations)')
    st.sidebar.text('Number of evaluations to curricular units in the 1st semester')

    st.sidebar.header('Curricular units 1st semester (approved)')
    st.sidebar.text('Number of curricular units approved in the 1st semester')

    st.sidebar.header('Curricular units 1st semester (grade)')
    st.sidebar.text('Grade average in the 1st semester (between 0 and 20)')

    st.sidebar.header('Curricular units 2nd semester (credited)')
    st.sidebar.text('Number of curricular units credited in the 2nd semester')

    st.sidebar.header('Curricular units 2nd semester (enrolled)')
    st.sidebar.text('Number of curricular units enrolled in the 2nd semester')

    st.sidebar.header('Curricular units 2nd semester (evaluations)')
    st.sidebar.text('Number of evaluations to curricular units in the 2nd semester')

    st.sidebar.header('Curricular units 2nd semester (approved)')
    st.sidebar.text('Number of curricular units approved in the 2nd semester')

    st.sidebar.header('Curricular units 2nd semester (grade)')
    st.sidebar.text('Grade average in the 2nd semester (between 0 and 20)')

    st.sidebar.header('GDP')
    st.sidebar.text('GDP (between -5 and 5)')


def predictResult(user_input, model, scaler, label_encoder):

    input_array = np.array([user_input[key] for key in user_input])

    data = {
        'Marital status': input_array[0],
        'Application mode': input_array[1],
        'Application order': input_array[2],
        'Course': input_array[3],
        'Daytime/evening attendance\t': input_array[4],
        'Previous qualification': input_array[5],
        'Previous qualification (grade)': input_array[6],
        'Admission grade': input_array[7],
        'Displaced': input_array[8],
        'Debtor': input_array[9],
        'Tuition fees up to date': input_array[10],
        'Gender': input_array[11],
        'Scholarship holder': input_array[12],
        'Age at enrollment': input_array[13],
        'Curricular units 1st sem (credited)': input_array[14],
        'Curricular units 1st sem (enrolled)': input_array[15],
        'Curricular units 1st sem (evaluations)': input_array[16],
        'Curricular units 1st sem (approved)': input_array[17],
        'Curricular units 1st sem (grade)': input_array[18],
        'Curricular units 2nd sem (credited)': input_array[19],
        'Curricular units 2nd sem (enrolled)': input_array[20],
        'Curricular units 2nd sem (evaluations)': input_array[21],
        'Curricular units 2nd sem (approved)': input_array[22],
        'Curricular units 2nd sem (grade)': input_array[23],
        'GDP': input_array[24],
    }

    #st.write(data)

    df = pd.DataFrame(data, index=[0])

   # st.write(df)
    
    #input_array = input_array.reshape(1, -1)

    #st.write(input_array)

    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))


    #trying to pass this into the model doesn't work as expected
    input_array = preprocessor.transform(df)

    st.write(input_array.shape)

    prediction = model.predict(input_array)

    return prediction

def displayResult(prediction):
    if prediction == 0:
        components.html(
            """
            <h1 style="color: black; font-size: 28px; text-align: center; font-weight: 600; font-family: sans-serif">The student is most likely going to be a dropout</h1>
            """,
            height=50,
        )
    elif prediction == 1:
        components.html(
            """
            <h1 style="color: black; font-size: 28px; text-align: center; font-weight: 600; font-family: sans-serif">The student is most likely enrolled</h1>
            """,
            height=50,
        )
    else:
        components.html(
            """
            <h1 style="color: black; font-size: 28px; text-align: center; font-weight: 600; font-family: sans-serif">The student is most likely going to be a graduate</h1>
            """,
            height=50,
        )

def main():
    loaded_model = loadModel()
    displayHeader()
    user_input = getUserInput()
    st.markdown(
    """<style>
    div[data-testid="stButton"] {
        text-align: center;
        margin-top: 25px;
    }
    div[data-testid="stMarkdownContainer"] {
     =
    }
    </style>""",
    unsafe_allow_html=True,
    )
    if st.button('Predict'):
        if any(value == '' for value in user_input.values()):
            st.error('Please fill in all the required fields')
        else:
            scaler = pickle.load(open('preprocessor.pkl', 'rb'))
            label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

            prediction = predictResult(user_input, loaded_model, scaler, label_encoder)
            displayResult(prediction)

    displayKeySidebar()

if __name__ == "__main__":
    main()



