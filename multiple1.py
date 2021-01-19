import streamlit as st
import pandas as pd
import numpy as np
#import plotly.express as px
#from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)



pickle_in = open('lin_mult.pkl', 'rb')
classifier = pickle.load(pickle_in)

df=pd.read_csv('cwurData.csv')
rad=st.sidebar.radio("HOME",["University Rank Prediction","Data Analysis","Data Summary"])


if rad=="University Rank Prediction":

    import base64

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return

    set_png_as_page_bg('im2.png')













    

    

    st.title('University Rank Prediction')

    score= st.number_input("University score :")

    qoe= st.number_input("Quality of education score:")

    ae= st.number_input("Alumini employment score:")

    qof= st.number_input("Quality of faculty score:")

    pub= st.number_input("Publications score :")

    inf= st.number_input("Influence score :")

    cit = st.number_input("Citations score :")
    patents = st.number_input("Patents score :")


    submit = st.button('World_Rank_Predict')

    if submit:
        prediction = classifier.predict([[score,qoe,ae,qof,pub,inf,cit,patents]])
        if round(prediction[0,0])<=0:
            st.write('world rank of your univeristy is',1)
        else:
            
            st.write('World rank of your univeristy is',round(prediction[0,0]))

##    submit = st.button('World_Rank_Predict')

##    if submit:
##        prediction = classifier.predict([[score,qoe,ae,qof,pub,inf,cit,patents]])
##        if round(prediction[0,1])<=0:
##            st.write('World rank of your univeristy is',1)
##        else:
##            
##            st.write('World rank of your univeristy is',round(prediction[0,1]))











if rad=="Data Analysis":

    st.title('Data Analysis')

    y=st.selectbox("SELECT",['heatmap','quality_of_education vs score','alumni_employment vs score','quality_of_faculty vs score','publications vs score',
                             'influence vs score','citations vs score', 'patents vs score',


                             'world_rank_histogram', 'national_rank_histogram','alumni_employment_histogram','quality_of_faculty_histogram',
                              'publications_histogram','influence_histogram','citations_histogram','patents_histogram','score_histogram'

        ,'national_rank vs quality_of_education','national_rank vs alumni_employment','national_rank vs quality_of_faculty',
                             'national_rank vs publications','national_rank vs influence','national_rank vs citations','national_rank vs patents','national_rank vs score',

                'world_rank vs quality_of_education','world_rank vs alumni_employment','world_rank vs quality_of_faculty',
                             'world_rank vs publications','world_rank vs influence','world_rank vs citations','world_rank vs patents','world_rank vs score' ])


    if y=='heatmap':

        corr =df.corr()
 

        sns.heatmap(corr)
        st.pyplot()

    elif y=='national_rank vs quality_of_education':

        sns.relplot(x='quality_of_education',y='national_rank',data=df,kind='line')
        st.pyplot()
    elif y=='national_rank vs alumni_employment':

        sns.relplot(x='alumni_employment',y='national_rank',data=df,kind='line')
        st.pyplot()
    elif y=='national_rank vs quality_of_faculty':

        sns.relplot(x='quality_of_faculty',y='national_rank',data=df,kind='line')
        st.pyplot()
    elif y=='national_rank vs publications':

        sns.relplot(x='publications',y='national_rank',data=df,kind='line')
        st.pyplot()

    elif y=='national_rank vs influence':

        sns.relplot(x='influence',y='national_rank',data=df,kind='line')
        st.pyplot()
    elif y=='national_rank vs citations':

        sns.relplot(x='citations',y='national_rank',data=df,kind='line')
        st.pyplot()

    elif y=='national_rank vs patents':

        sns.relplot(x='patents',y='national_rank',data=df,kind='line')
        st.pyplot()
    elif y=='national_rank vs score':

        sns.relplot(x='score',y='national_rank',data=df,kind='line')
        st.pyplot()


    elif y=='world_rank_histogram':

        sns.distplot(df.world_rank)
        st.pyplot()

    elif y=='national_rank_histogram':

        sns.distplot(df.national_rank)
        st.pyplot()

    elif y=='alumni_employment_histogram':

        sns.distplot(df.alumni_employment)
        st.pyplot()
    elif y=='quality_of_faculty_histogram':

        sns.distplot(df.quality_of_faculty)
        st.pyplot()

    elif y=='publications_histogram':

        sns.distplot(df.publications)
        st.pyplot()
    elif y=='influence_histogram':

        sns.distplot(df.influence)
        st.pyplot()

    elif y=='citations_histogram':

        sns.distplot(df.citations)
        st.pyplot()
    elif y=='patents_histogram':

        sns.distplot(df.patents)
        st.pyplot()
    elif y=='score_histogram':

        sns.distplot(df.score)
        st.pyplot()



    elif y=='heatmap':

        corr =df.corr()
 

        sns.heatmap(corr)
        st.pyplot()

    elif y=='alumni_employment vs score':

        sns.jointplot(df.alumni_employment, df.score)
        st.pyplot()

    elif y=='quality_of_education vs score':

        sns.jointplot(df.quality_of_education, df.score)
        st.pyplot()

    elif y=='quality_of_faculty vs score':

        sns.jointplot(df.quality_of_faculty, df.score)
        st.pyplot()

    elif y=='publications vs score':

        sns.jointplot(df.publications, df.score)
        st.pyplot()
    elif y=='influence vs score':

        sns.jointplot(df.influence, df.score)
        st.pyplot()
    elif y=='citations vs score':

        sns.jointplot(df.citations, df.score)
        st.pyplot()
    elif y=='patents vs score':

        sns.jointplot(df.patents, df.score)
        st.pyplot()


    elif y=='world_rank vs quality_of_education':

        sns.relplot(x='quality_of_education',y='world_rank',data=df,kind='line')
        st.pyplot()
    elif y=='world_rank vs alumni_employment':

        sns.relplot(x='alumni_employment',y='world_rank',data=df,kind='line')
        st.pyplot()
    elif y=='world_rank vs quality_of_faculty':

        sns.relplot(x='quality_of_faculty',y='world_rank',data=df,kind='line')
        st.pyplot()
    elif y=='world_rank vs publications':

        sns.relplot(x='publications',y='world_rank',data=df,kind='line')
        st.pyplot()

    elif y=='world_rank vs influence':

        sns.relplot(x='influence',y='world_rank',data=df,kind='line')
        st.pyplot()
    elif y=='world_rank vs citations':

        sns.relplot(x='citations',y='world_rank',data=df,kind='line')
        st.pyplot()

    elif y=='world_rank vs patents':

        sns.relplot(x='patents',y='world_rank',data=df,kind='line')
        st.pyplot()
    elif y=='world_rank vs score':

        sns.relplot(x='score',y='world_rank',data=df,kind='line')
        st.pyplot()


  


  
        

    

if rad=="Data Summary":
    st.title('Data Sheet Overview')
    st.text('Here you can find the selected data set')

    st.dataframe(df)

    st.text('The summary of the data set is added below')

    summary=df.describe()
    st.dataframe(summary)


    

    
    
        
        
    

