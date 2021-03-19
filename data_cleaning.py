import pandas as pd
from my_package import clean_course_num


def clean_all_dataset():
    # Clean the adult dataset
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
               "native-country", ">50K"]
    adult = pd.read_csv('data/adult.data', names=columns)
    adult['>50K'] = adult['>50K'].apply(lambda x: 1 if x.strip() != "<=50K" else -1)
    encoded = pd.get_dummies(adult, drop_first=True)

    with open("data/adult_clean.csv", 'w') as file:
        file.write(encoded.to_csv())
        file.close()

    # Clean the cape dataset
    cape = pd.read_csv("data/CAPE_all.csv")
    cape = cape.assign(course_code=cape['Course'].apply(lambda course: str(course).split('-')[0][:-1]))
    cape = cape.assign(department=cape["course_code"].apply(lambda code: str(code).split()[0]))
    cape = cape.assign(course_num=cape["course_code"].apply(
        lambda code: str(code).split()[1]
        if len(str(code).split()) == 2 else code))
    cape = cape.assign(course_description=cape['Course'].apply(
        lambda course: str(course).split('-')[1] if len(str(course).split('-')) == 2 else course))
    grade = cape[['department', 'course_num', 'Term', 'Study Hrs/wk', 'Avg Grade Expected', 'Avg Grade Received']][
        (cape["Avg Grade Expected"].notna()) & (cape["Avg Grade Received"].notna())
    ]
    grade = grade.assign(
        GPA_Expected=grade['Avg Grade Expected'].apply(lambda grade_: float(grade_.split()[1][1:-1])),
        GPA_Received=grade['Avg Grade Received'].apply(lambda grade_: float(grade_.split()[1][1:-1])),
        letter_Recieved=grade['Avg Grade Received'].apply(lambda grade_: grade_.split()[0])
    )
    grade["GPA_Received"] = grade["GPA_Received"].apply(lambda grade_: 1 if grade_ > 3.2 else -1)
    grade = grade.drop(columns=['Avg Grade Expected', 'Avg Grade Received', 'letter_Recieved'])
    grade['is_upper'] = grade['course_num'].apply(clean_course_num)
    grade = grade.drop(columns=['course_num'])
    grade_encoded = pd.get_dummies(grade, drop_first=True)
    with open('data/cape_clean.csv', 'w') as file:
        file.write(grade_encoded.to_csv())
        file.close()

    # Clean the COV dataset
    columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
               "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
               "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
               "Horizontal_Distance_To_Fire_Points"] + \
              ["Wilderness_Area_" + str(i) for i in range(4)] + \
              ["Soil_Type_" + str(i) for i in range(40)] + \
              ['Cover_Type']
    cov_raw = pd.read_csv("data/covtype.data.gz", names=columns)
    cov_raw['Cover_Type'] = cov_raw['Cover_Type'].apply(
        lambda type_num: 1 if type_num == 7 else -1)
    with open('data/cover_clean.csv', 'w') as file:
        file.write(cov_raw.to_csv())
        file.close()

    # Clean the Letter dataset
    columns = ['letter', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar',
               'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
               'y-ege', 'yegvx']
    letter_raw = pd.read_csv("data/letter-recognition.data", names=columns)
    letter_p1 = letter_raw.assign(
        letter=letter_raw['letter'].apply(lambda letter: 1 if letter == 'O' else -1)
    )
    positive_class = [chr(i) for i in range(ord("A"), ord("M") + 1)]
    letter_p2 = letter_raw.assign(
        letter=letter_raw['letter'].apply(lambda letter: 1 if letter in positive_class else -1)
    )
    with open("data/letter_clean_p1.csv", 'w') as file:
        file.write(letter_p1.to_csv())
        file.close()
    with open("data/letter_clean_p2.csv", 'w') as file:
        file.write(letter_p2.to_csv())
        file.close()
