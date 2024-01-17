# -*- coding: utf-8 -*-
"""
ClinLab.ipynb
"""


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import csv
import plotly.express as px
import chart_studio.plotly as pxchart
import numpy as np
from sklearn import linear_model, metrics
from sklearn.datasets import *
from sklearn.metrics import r2_score
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import io
import itertools
import scipy
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import contingency
from scipy.stats.contingency import relative_risk
import scikit_posthocs as sp
import math
from sigfig import round
import random
import base64
from PIL import Image




st.title("STAT")
st.markdown("_How quickly do want your analysis results? - **STAT** is the answer._")
st.text("  ")
st.text("  ")
st.text("  ")
tab6, tab2, tab3, tab4, tab5 = st.tabs(["Guide","Descriptive and Normality Testing", "Boxplot", "Comparing Groups", "Correlation"])

with tab6:
    
 

    # pdf1 = Image.open('C:/Users/carina.t/Dropbox/hello_world/20230810 front page annotations1024_1.png')
    pdf1 = Image.open('images/20230810 front page annotations1024_1.png')
             # Display the image
    st.image(pdf1, use_column_width=False)
    
    # pdf2 = Image.open('C:/Users/carina.t/Dropbox/hello_world/20230810 front page annotations1024_2.png')
    pdf2 = Image.open('images/20230810 front page annotations1024_2.png')
             # Display the image
    st.image(pdf2, use_column_width=False)

    # pdf3 = Image.open('C:/Users/carina.t/Dropbox/hello_world/20230810 front page annotations1024_3.png')
    pdf3 = Image.open('images/20230810 front page annotations1024_3.png')
             # Display the image
    st.image(pdf3, use_column_width=False)

with st.sidebar:

   st.header('Data Uploader')
   st.write('To use this website, prepare a .csv or .xlsx file according to tutorial.')

   
   if st.checkbox("Use Example Data", value=True):
       st.write ("Load Completed: Currently working with Example Data")
    
       df = pd.read_csv('Demo_Dataset_For_Clinlab.csv')
    
       df['event']= np.random.choice(['Yes','No'], len(df), p=[.7,.3])
       df['health_status']= np.random.choice(['Healthy','At Risk'], len(df), p=[.6,.4])

   
   else: 
    cleaningfile = st.file_uploader('')
    if cleaningfile is not None:
        filename=cleaningfile.name
        
        if ".csv" in filename:
            df = pd.read_csv(cleaningfile)
        elif ".xls" in filename:    
            df = pd.read_excel(cleaningfile)
        else: 
            st.write('Please provide CSV or Excel files only.')

    else:
        st.stop()

   #filter dataframe section

   st.header("Data Filter")
   st.write(
       """Use this data filter if you're only analysing a subgroup; this filter is applied to all tests in the tabs on the right
       """
   )
    
   @st.cache_data(experimental_allow_widgets=True)
   def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
       modify = st.checkbox("Do you need to filter your data?")
       st.caption ("Leave unchecked to use whole dataset")

       if not modify:
           return df

       #make a copy for the dataframe so it would not affect the original df
       df = df.copy()

   # Try to convert datetimes into a standard format (datetime, no timezone)
       for col in df.columns:

           if is_object_dtype(df[col]):
               try:
                   df[col] = pd.to_datetime(df[col])
               except Exception:
                   pass

           if is_datetime64_any_dtype(df[col]):
               df[col] = df[col].dt.tz_localize(None)


       modification_container = st.container()

       with modification_container:
           to_filter_columns = st.multiselect("Filter dataframe on", df.columns)

           for column in to_filter_columns:
               left, right = st.columns((1, 20))

               if is_datetime64_any_dtype(df[column]):
                   user_date_input = right.date_input(
                       f"Values for {column}",
                       value=(
                           df[column].min(),
                           df[column].max(),
                       ),
                   )
                   if len(user_date_input) == 2:
                       user_date_input = tuple(map(pd.to_datetime, user_date_input))
                       start_date, end_date = user_date_input
                       df = df.loc[df[column].between(start_date, end_date)]
               
               elif is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                   user_cat_input = right.multiselect(
                       f"Values for {column}",
                       df[column].unique(),
                       default=list(df[column].unique()),
                   )
                   df = df[df[column].isin(user_cat_input)]

               elif is_numeric_dtype(df[column]):
                   _min = float(df[column].min())
                   _max = float(df[column].max())
                   step = (_max - _min) / 100
                   user_num_input = right.slider(
                       f"Values for {column}",
                       min_value=_min,
                       max_value=_max,
                       value=(_min, _max),
                       step=step,
                   )
                   df = df[df[column].between(*user_num_input)]
               
               else:
                   user_text_input = right.text_input(
                       f"Substring or regex in {column}",
                   )
                   if user_text_input:
                       df = df[df[column].astype(str).str.contains(user_text_input)]

       return df

   category_features = []
   threshold = 5
   for each in df.columns:
       if df[each].nunique() < threshold:
          category_features.append(each)

   for each in category_features:
       df[each] = df[each].astype('category')

   cleandata = filter_dataframe(df)
   st.dataframe(cleandata)

   @st.cache_data
   def convert_df(df):
       # IMPORTANT: Cache the conversion to prevent computation on every rerun
       return df.to_csv().encode('utf-8')

   csv = convert_df(cleandata)

   st.download_button(
       label="Download data as CSV",
       data=csv,
       file_name='filtered_dataframe.csv',
       mime='text/csv',
   )




with tab2:

        st.header('Histogram and Descriptive Statistics')

        st.caption ("A histogram helps you to visualise the distribution of your data; change the number of bins to optimise")

        #select number of bins for histogram
        number_bins = st.slider(label='Choose number of bins here', min_value=3, max_value=50, step=1)

        #select variable for histogram and descriptive statistics
        selected_histo_var = st.selectbox('What do want the x variable to be?', cleandata._get_numeric_data().columns)

        #histogram
        if st.checkbox("Optional: Change axis labels? ", value=False):
            ylabel = st.text_input("Input New y-label", 'Text')
            xlabel = st.text_input("Input New x-label", 'Text')
            fig = px.histogram(cleandata, nbins = number_bins, x = cleandata[selected_histo_var], labels={"value": xlabel, "count": ylabel}
                ).update_layout(
                xaxis_title=xlabel, yaxis_title=ylabel
                )
        else:
            fig = px.histogram(cleandata, nbins = number_bins, x = cleandata[selected_histo_var])
        
        # Save image of Histogram before displaying on Streamlit - so it can be saved
        fn = 'histogram.eps'
        plt.savefig(fn,format='eps')
        st.plotly_chart(fig)

        with open(fn, "rb") as img:
            btn = st.download_button(
            label="Download Histogram",
            data=img,
            file_name=fn,
            mime="application/octet-stream"
            )

        #descriptive statistics if numeric data:
        selected_descr_var = st.selectbox('Which Variable would you like to generate Descriptive Statistics for?', cleandata.columns)

        #descriptive statistics if numeric data:
        if is_numeric_dtype(cleandata[selected_descr_var]):
            description = cleandata[selected_descr_var].describe()
            st.write (description)
            st.caption ('Note that 50th percentile = median.')

        else: 
            summary = cleandata[selected_descr_var].value_counts()
            percents_summary = cleandata[selected_descr_var].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            cat_summary = pd.concat([summary,percents_summary], axis=1, keys=['Counts', 'Proportions'])
            st.dataframe (cat_summary)

        st.header ("Which of my continuous variables are normally distributed?")
        st.caption("Test conducted using Shapiro-Wilk test for continuous variables only.")

        # normality testing for all numeric variables
        for variable in cleandata.columns:
            if is_numeric_dtype(cleandata[variable]):
                statistic = cleandata.filter (items = [variable])
                statistic.dropna(inplace=True)
                if len(statistic) > 3:
                    normality = stats.shapiro(statistic)
                    normality_stat = round(normality.statistic,5)
                    normality_p = normality.pvalue
                    if normality_p < 0.0001:
                        normality_p_display = "<0.0001"
                    else:
                        normality_p_display = round(normality_p,5)


                    if normality_p <0.05:
                        is_column_normal = "Not Normal Distribution"
                        st.write (f'{variable} : **:red[{is_column_normal}]** [ W: {normality_stat} , p-value: {normality_p_display} ]')
                    else:
                        is_column_normal = "Normal Distribution"
                        st.write (f'{variable} : :green[{is_column_normal}]** [ W: {normality_stat} , p-value: {normality_p_display} ]')

                else: 
                    st.write(f'{variable} has less than 3 data points, normality test cannot be performed.')

        st.caption ('When p-value of Shapiro-Wilk test is < 0.05:  \n - This column of data is not normally distributed.  \n - Non-parametric tests are recommended for downstream analysis.  ')
        st.caption ('When p-value of Shapiro-Wilk test is > 0.05: \n - Data has a normal distribution and parametric tests can be used. ')




with tab3:
    dataset = cleandata


    # Swarm on Boxplot
    st.header("Swarm on Boxplot")
    sns.set_style("white")


   # Select more than one categorical columns and concatenate into a new column for x
    selected_x_var = st.multiselect(
        'What do you want the x- variable to be? Note that this must be a categorical variable, and that any NA must be first removed with Data Filter function',
        dataset.select_dtypes(exclude=["number"]).columns)

   # Select one continuous column for y
    selected_y_var = st.selectbox('What do you want the y-variable to be? Note that this must be a continuous variable; NA will be automatically removed', dataset._get_numeric_data().columns)

    if st.checkbox("Optional: Color by group?", value=False):
        col_by = st.selectbox('Hue?', dataset.select_dtypes(exclude=["number"]).columns)

        #Converting Integer Categorical variables to String
        dataset[selected_x_var] = dataset[selected_x_var].astype(str)
        dataset["x"]= dataset[selected_x_var].apply('-'.join, axis=1)


        #Plot the swarm/boxplot
        fig, ax = plt.subplots()
        ax = sns.swarmplot(x="x", y=selected_y_var, data=dataset, hue=col_by)
        ax = sns.boxplot(x="x", y=selected_y_var, data=dataset, whis=[5,95], showfliers = False,
            showcaps=True,boxprops={'facecolor':'None'})
        if st.checkbox("Optional: Change axis labels?", value=False):
            ylabel = st.text_input(
                "Enter y-label", 'Text'
                )
            xlabel = st.text_input(
                "Enter x-label", 'Text'
                )
            ax.set(xlabel=xlabel, ylabel=ylabel)
            fn2 = 'swarm.eps'
            plt.savefig(fn2,format='eps')
            st.pyplot(fig)
            st.write('For box-whisker plots, line indicates median, upper and lower boundaries of boxes indicate upper and lower quartile respectively, and whiskers indicate the 5th and 95th percentile.')

            with open(fn2, "rb") as img2:
                btn = st.download_button(
                    label="Download Boxplot",
                    data=img2,
                    file_name=fn2,
                    mime="application/octet-stream"
                )

        else:
            fn2 = 'swarm.eps'
            plt.savefig(fn2,format='eps')
            st.pyplot(fig)
            st.write('For box-whisker plots, line indicates median, upper and lower boundaries of boxes indicate upper and lower quartile respectively, and whiskers indicate the 5th and 95th percentile.')

            with open(fn2, "rb") as img2:
                    btn = st.download_button(
                        label="Download Boxplot",
                        data=img2,
                        file_name=fn2,
                        mime="application/octet-stream"
                    )

    else:
        #Converting Integer Categorical variables to String
        dataset[selected_x_var] = dataset[selected_x_var].astype(str)
        dataset["x"]= dataset[selected_x_var].apply('-'.join, axis=1)


    #Plot the swarm/boxplot
        fig, ax = plt.subplots()
        ax = sns.swarmplot(x="x", y=selected_y_var, data=dataset)
        ax = sns.boxplot(x="x", y=selected_y_var, data=dataset, whis=[5,95], showfliers = False,
                showcaps=True,boxprops={'facecolor':'None'})

        if st.checkbox("Optional: Change axis labels?", value=False):
            ylabel = st.text_input(
                "Enter y-label", 'Text'
                )
            xlabel = st.text_input(
                "Enter x-label", 'Text'
                )
            ax.set(xlabel=xlabel, ylabel=ylabel)
            # Save image of Box Plot before displaying on Streamlit - so it can be saved
            fn2 = 'swarm.eps'
            plt.savefig(fn2,format='eps')
            st.pyplot(fig)
            st.write('For box-whisker plots, line indicates median, upper and lower boundaries of boxes indicate upper and lower quartile respectively, and whiskers indicate the 5th and 95th percentile.')

            with open(fn2, "rb") as img2:
                btn = st.download_button(
                    label="Download Boxplot",
                    data=img2,
                    file_name=fn2,
                    mime="application/octet-stream"
                )

        else:
            # Save image of Box Plot before displaying on Streamlit - so it can be saved
            fn2 = 'swarm.eps'
            plt.savefig(fn2,format='eps')
            st.pyplot(fig)
            st.write('For box-whisker plots, line indicates median, upper and lower boundaries of boxes indicate upper and lower quartile respectively, and whiskers indicate the 5th and 95th percentile.')

            with open(fn2, "rb") as img2:
                btn = st.download_button(
                    label="Download Boxplot",
                    data=img2,
                    file_name=fn2,
                    mime="application/octet-stream"
                )

with tab4:
     #blank


     scatters_df = cleandata
     st.caption("If analysis takes very long to load, try removing the variables selected in the Correlation tab.")

     st.header("Unpaired T-Test for Continuous Variables")
     st.caption ("T-test is parametric. Mann-Whitney U test is non-parametric. Both test the difference between TWO groups. Note that T-test performed does not assume equal population variance.")

     selected_var = st.selectbox('What is the variable to be tested? Note that this has to be a continuous variable. ', scatters_df._get_numeric_data().columns)
     selected_categorical_var = st.selectbox('What is the variable used to separate the groups? ', scatters_df.select_dtypes(exclude=["number"]).columns )

     var = set(scatters_df.columns)
     pop = list(set(scatters_df[selected_categorical_var]))
    #  print(scatters_df[selected_categorical_var])

     if selected_categorical_var in var:
         selected_pop_x = st.selectbox('What is the test group?', pop)
         selected_pop_y = st.selectbox('What is the control group?', pop)

         dataframe_filter2 = scatters_df[[selected_var,selected_categorical_var]]
         dataframe_filter = dataframe_filter2.dropna(axis = 'rows')
         st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', dataframe_filter.shape[0], 'of', dataframe_filter2.shape[0])
         x_var = np.array( dataframe_filter[selected_var] )[ dataframe_filter[selected_categorical_var] == selected_pop_x]
         y_var = np.array( dataframe_filter[selected_var] )[ dataframe_filter[selected_categorical_var] == selected_pop_y]


        #  print(x_var)
        #  print(y_var)
         _, pnorm_mann = mannwhitneyu(x_var, y_var) #method="asymptotic")

         if pnorm_mann < 0.0001:
            pnorm_mann = "<0.0001"
         else:
            pnorm_mann = round(pnorm_mann,5)

         _, pnorm_ttest =  ttest_ind(x_var, y_var, equal_var = False) # method="asymptotic") #Assuming unequal variances

         if pnorm_ttest < 0.0001:
            pnorm_ttest = "<0.0001"
         else:
            pnorm_ttest = round(pnorm_ttest,5)

         st.write(f'T-test p-value: {pnorm_ttest}')
         st.write(f'Mann-Whitney U p-value: {pnorm_mann}')


     st.header("Paired T-Test for Continuous Variables")
     st.caption ("Paired T-Test is parametric. Wilcoxon Signed Rank test is non-parametric. Both test the difference between TWO groups. \n Zero-differences between two pairs for Wilcoxon Rank Sum test would be discarded.")

     paired_selected_var = st.selectbox('What is the test group? (eg. Post-Treatment) ', scatters_df._get_numeric_data().columns)

     scatters_df_without_selected = scatters_df.drop(paired_selected_var, axis=1)
     paired_selected_control_var = st.selectbox('What is the control group? (eg. Baseline, Pre-Treatment) ', scatters_df_without_selected._get_numeric_data().columns)

     paired_var = set(scatters_df.columns)


     if paired_selected_control_var in paired_var:

         dataframe_filter3 = scatters_df[[paired_selected_var,paired_selected_control_var]]
         dataframe_filter4 = dataframe_filter3.dropna(axis = 'rows')
         st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', dataframe_filter4.shape[0], 'of', dataframe_filter3.shape[0])

         paired_x_var = scatters_df[paired_selected_var]
         paired_y_var = scatters_df[paired_selected_control_var]


        #  print(paired_x_var)
        #  print(paired_y_var)
         _, pnorm_pairedt = stats.ttest_rel(paired_y_var, paired_x_var) #method="asymptotic")

         if pnorm_pairedt < 0.0001:
            pnorm_pairedt = "<0.0001"

         from scipy.stats import wilcoxon
         _, pnorm_wilcoxon =  wilcoxon(paired_y_var, paired_x_var)

         if pnorm_wilcoxon < 0.0001:
            pnorm_wilcoxon = "<0.0001"

         st.write(f'Paired T-test p-value: {pnorm_pairedt}')
         st.write(f'Wilcoxon Ranked Sum Test p-value: {pnorm_wilcoxon}')




         st.header("One-way ANOVA for Continuous Variables")
         st.caption('This comparison tests the difference between 3 groups or more. One-way ANOVA is parametric. Kruskal-Wallis H test is non-parametric. If there are only 2 groups, T-test will be performed instead.')
         anovatype = st.selectbox(
            'Which test would you want to use?',
            ("One-way ANOVA", "Kruskal-Wallis H-test"))

         if anovatype == 'One-way ANOVA':
             st.write("You selected: One-way ANOVA")
         elif anovatype == 'Kruskal-Wallis H-test':
             st.write("You selected: Kruskal-Wallis H-test")

         selected_var1 = st.selectbox('What is the variable to be tested? Note that this has to be a continuous variable.', scatters_df._get_numeric_data().columns)
         selected_categorical_var1 = st.selectbox('What is the group used to separate the variable selected?', scatters_df.select_dtypes(exclude=["number"]).columns)
         pop = list(set(scatters_df[selected_categorical_var1]))

         dataframe_filter3 = scatters_df[[selected_var1,selected_categorical_var1]]
         dataframe_filter4 = dataframe_filter3.dropna(axis = 'rows')
         st.write('Number of datapoints retained after removing NaN:', dataframe_filter4.shape[0], 'of', dataframe_filter3.shape[0])

         to_one_way = {}
         for v in pop:
             
                to_one_way[v] = list(dataframe_filter4[selected_var1][np.array(dataframe_filter4[selected_categorical_var1] == v) ])
                to_one_way[v] = np.array(to_one_way[v])[ np.array(to_one_way[v]) < 1000000000000000000]
             
                
         if len(to_one_way) > 2:
            if anovatype == 'One-way ANOVA':

                _, pnorm_one = f_oneway( *to_one_way.values())
                if pnorm_one < 0.0001:
                    pnorm_one = "<0.0001"
                    
                else:
                    pnorm_one = round(pnorm_one,5)
                st.write(f'One-way Anova p value: {pnorm_one}')
                
                

            elif anovatype == 'Kruskal-Wallis H-test':
                 _, pnorm_kw = stats.kruskal( *to_one_way.values())
                 if pnorm_kw < 0.0001:
                    pnorm_kw = "<0.0001"
                   
                 else:
                    pnorm_kw = round(pnorm_kw,5)

                 st.write(f'Kruskal-Wallis H-test p value: {pnorm_kw}')
                 

         elif len(to_one_way) == 2:

            if anovatype == 'One-way ANOVA':
                _, pnorm_one = f_oneway( *to_one_way.values())
                if pnorm_one < 0.0001:
                    pnorm_one = "<0.0001"
                else:
                    pnorm_one = round(pnorm_one,5)
                st.write(f'T-test p value: {pnorm_one}')
                st.caption(f'NOTICE: As <{selected_var1}> have less than two groupings when grouped by <{selected_categorical_var1}>, T-test is done instead of ANOVA. ')

            elif anovatype == 'Kruskal-Wallis H-test':
                _, pnorm_kw = stats.kruskal( *to_one_way.values())
                if pnorm_kw < 0.0001:
                    pnorm_kw = "<0.0001"
                   
                else:
                    pnorm_kw = round(pnorm_kw,5)

                st.write(f'Mann Whitney p value: {pnorm_kw}')
                st.caption(f':red[NOTICE:] As <{selected_var1}> have less than two groupings when grouped by <{selected_categorical_var1}>, Mann Whitney U Test is done instead of Kruskal-Wallis H-test. ')
         else:
            st.write(f'NULL: Selected variable <{selected_var1}> have less than two groupings when grouped by <{selected_categorical_var1}>.')
            #st.write('test4')

         st.caption('If p-value < 0.05, please proceed with post-hoc test to conduct a set of pairwise comparisons to determine which groups are significantly different from the other. ')
                
         if st.checkbox('Would you like to do a Post-hoc Test?'):
             
             if anovatype == 'One-way ANOVA':
                test100 = sp.posthoc_ttest(dataframe_filter4,val_col = selected_var1, group_col = selected_categorical_var1, p_adjust = 'bonferroni')

             elif anovatype == 'Kruskal-Wallis H-test':
                test100 = sp.posthoc_mannwhitney(dataframe_filter4,val_col = selected_var1, group_col = selected_categorical_var1, p_adjust = 'bonferroni')

             st.write(" ")
             st.write(" ")
             st.write("p-values of Post-hoc test - with Multiple Test Correction - Bonferroni:")
             st.write(test100)


         else:
             st.write('Click to start analysis')

         st.header("Chi-squared Test")

         st.caption ("Chi-squared test is appropriate when comparing proportions between groups")

         selected_var2 = st.selectbox('What is the variable to be compared? Note that this has to be a categorical variable', scatters_df.select_dtypes(exclude=["number"]).columns )
         selected_categorical_var2 = st.selectbox(' What is the variable used to separate the groups?', scatters_df.select_dtypes(exclude=["number"]).columns )


         x_var2 = np.array( scatters_df[selected_var2] )
         y_var2 = np.array( scatters_df[selected_categorical_var2] )


         crosstab = pd.crosstab([x_var2], [y_var2])
         # Calculate the percentages of each cell
         crosstab_perc = crosstab.apply(lambda x: x/x.sum(), axis=1)
         # Format the percentages
         crosstab_perc = crosstab_perc.applymap(lambda x: '{:.1%}'.format(x))

         chi2, p, dof, expected = stats.chi2_contingency(crosstab)
         pnorm_chisq = p
         
         if pnorm_chisq < 0.0001:
            pnorm_chisq = "<0.0001"
         else:
            pnorm_chisq = round(pnorm_chisq,5)
         st.write(" ")
         st.caption('Chi-squared Test of Independence requires n =/> 5 in each observed value cell.')
         st.write(" ")

         st.write("Crosstab Row represents: ",selected_var2," \n Crosstab Column represents: ",selected_categorical_var2)
         col1, col2 = st.columns(2)
         with col1:
             st.dataframe(crosstab)

         with col2:
             st.dataframe(crosstab_perc)

         st.write(f'Chi-squared p-value: {pnorm_chisq}') 


         st.header("Relative Risk and Odds Ratio")

         st.caption("Relative risk calculation is utilised in Cohort studies.  Odds Ratio is utilised in Case-Control / Cross-Sectional studies. \n Ensure that only categorical variables with two groups are selected - i.e. Yes/No, Young/Old, At Risk/Healthy.")
 
         st.write(" ")

         
         image = Image.open('images/Relative_Risk_Odds_Ratio_Picture.png')
             # Display the image
         st.image(image, caption='Relative Risk Formula', use_column_width=True)

         

         independent_var = st.selectbox('Group', scatters_df.select_dtypes(exclude=["number"]).columns )
         
         variables2 = set(scatters_df.columns)
         pop_set3 = list(set(scatters_df[independent_var]))
         st.markdown(pop_set3)


         if independent_var in variables2:
            selected_pop_reference = st.selectbox('Which is the value in your selected grouping that represents reference group? Eg. Placebo, Did not attend Childcare', pop_set3)

         st.write(" ")
         st.write(" ")

         dependent_var = st.selectbox('Event', scatters_df.select_dtypes(exclude=["number"]).columns )

         pop_set2 = list(set(scatters_df[dependent_var]))
         st.markdown(pop_set2) #checkpoint

         st.write(" ")
         if independent_var in variables2:
            selected_pop_case = st.selectbox('Which is the value that represents that event did happened? Eg. Positive for Disease', pop_set2)


            dataframe_filter10 = scatters_df[[independent_var,dependent_var]]

            dataframe_filter11 = dataframe_filter10.dropna(axis = 'rows')
            st.write(" ")
            st.write(" ")
            st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', dataframe_filter11.shape[0], 'of', dataframe_filter10.shape[0])
            #st.write(dataframe_filter11) #checkpoint

            all_positive_characteristics = dataframe_filter11[dataframe_filter11[independent_var]!=selected_pop_reference]

            all_negative_characteristics = dataframe_filter11[(dataframe_filter11[independent_var]==selected_pop_reference)]
           

            exposed_cases_dataframe = all_positive_characteristics[(all_positive_characteristics[dependent_var]==selected_pop_case)]
          

            control_cases_dataframe = all_negative_characteristics[(all_negative_characteristics[dependent_var]==selected_pop_case)]
        
            
            control_noncases_dataframe = all_negative_characteristics[(all_negative_characteristics[dependent_var]!=selected_pop_case)]
           

            exposed_noncases_dataframe = all_positive_characteristics[(all_positive_characteristics[dependent_var]!=selected_pop_case)]
        


## PULLING AGGREGATE DATA FROM CROSSTAB TO DO RELATIVE RISK
         exposed_cases = len(exposed_cases_dataframe) #positive event for at-risk characteristics
         
          
         exposed_total = len(all_positive_characteristics) #total event for at-risk characteristics
       
         
         control_cases = len(control_cases_dataframe)  #positive event for reference characteristics

         control_total = len(all_negative_characteristics) #total events for reference characteristics
         

         control_noevent = len(control_noncases_dataframe) #total events for reference characteristics
       

         treatment_noevent = len(exposed_noncases_dataframe) #total events for reference characteristics
        

         relative_risk_result = relative_risk(exposed_cases, exposed_total, control_cases, control_total)
         rel_risk = relative_risk_result.relative_risk
         conf_intvl = relative_risk_result.confidence_interval(confidence_level = 0.95)
         low_conf_int = conf_intvl[0]
         high_conf_int = conf_intvl[1]


         if rel_risk < 0.0001:
            rel_risk = "<0.0001"
         else:
            rel_risk = round(rel_risk,5)

         if low_conf_int < 0.0001:
            low_conf_int = "<0.0001"
         else:
            low_conf_int = round(low_conf_int,5)

         if high_conf_int < 0.0001:
            high_conf_int = "<0.0001"
         else:
            high_conf_int = round(high_conf_int,5)

         st.write(" ")
         
         st.write(f'Relative Risk: {rel_risk}  \n Confidence Interval (95%):   Low - {low_conf_int}  ; High - {high_conf_int}')

         
         odds_ratio_result = ((exposed_cases*control_noevent)/(treatment_noevent*control_cases))

         low_conf_int_odd = math.exp((np.log(odds_ratio_result))-(1.96*(math.sqrt((1/exposed_cases)+(1/control_noevent)+(1/treatment_noevent)+(1/control_cases)))))
         high_conf_int_odd = math.exp((np.log(odds_ratio_result))+(1.96*(math.sqrt((1/exposed_cases)+(1/control_noevent)+(1/treatment_noevent)+(1/control_cases)))))


         if odds_ratio_result < 0.0001:
            odds_ratio_result = "<0.0001"
         else:
            odds_ratio_result = round(odds_ratio_result,5)

         if low_conf_int_odd < 0.0001:
            low_conf_int_odd = "<0.0001"
         else:
            low_conf_int_odd = round(low_conf_int_odd,5)

         if high_conf_int_odd < 0.0001:
            high_conf_int_odd = "<0.0001"
         else:
            high_conf_int_odd = round(high_conf_int_odd,5)

         st.write(" ")
         
         st.write(f'Odds Ratio: {odds_ratio_result}  \n Confidence Interval (95%):   Low - {low_conf_int_odd}  ; High - {high_conf_int_odd}')

with tab5:

   @st.cache_data
   def pearson_calculate_corr(df):
    dfcols = pd.DataFrame(columns=df.columns)
    correlation_r = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            correlation_r[r][c] = round(pearsonr(tmp[r], tmp[c])[0], 4)
    return correlation_r
       
   @st.cache_data
   def spearman_calculate_corr(df):
    dfcols = pd.DataFrame(columns=df.columns)
    correlation_r = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            correlation_r[r][c] = round(spearmanr(tmp[r], tmp[c])[0], 4)
    return correlation_r

   @st.cache_data
   def pearson_calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues

   @st.cache_data
   def spearman_calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(spearmanr(tmp[r], tmp[c])[1], 4)
    return pvalues

   df = cleandata
   if st.checkbox("Leave unchecked if you move to another Tab - as it will slow down analysis significantly.", value=False):
       
       st.header('Part 1: Correlation analysis')
       st.caption("Pearson correlation is only applicable to two variables that are linearly related; otherwise Spearman correlation should be used. In addition, Spearman correlation is recommended for non-parametric data. When in doubt, use Spearman correlation.")
   
       corrtype = st.selectbox(
       'What correlation test do you want to use?',
       ("pearson", "spearman"))

       if corrtype == 'pearson':
           st.write("You selected: Pearson's correlation")
       elif corrtype == 'spearman':
           st.write("You selected: Spearman's rank correlation")

       ### User selection of variables
       varlist = df.select_dtypes(include=['float64','int64']).columns
       filter = st.multiselect(
               'Select up to 10 variables for pairwise correlation analysis', varlist)
       if len(filter) >= 11:
           st.markdown('<p style="color:Red; font-size: 30px;">**CAUTION**: too many variables will significantly slow down computation!</p>', unsafe_allow_html=True)
       pass


   ### Sorting the list of selected variables, then filtering the dataset
       filter = sorted(filter)
       df_filter = df.filter(items=filter)

   ### Check for NaN
       if df_filter.isnull().sum().sum() >= 1:
           st.markdown('<p style="color:Orange;">**WARNING**: empty cells and/or null values (NaN) detected in dataset!</p>', unsafe_allow_html=True)
           st.write(df_filter.isnull().sum().sum(), 'invalid cells detected. Samples with invalid cells are removed from analysis.')
       pass

   ### Removing rows with NaN - to remove drop NA
       df_filter_droppedna = df_filter.dropna(axis='rows', how= 'any') #
       st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', df_filter_droppedna.shape[0], 'of', df_filter.shape[0])

       if len(filter) >= 2:
           if st.checkbox('Start correlation analysis!'):
               st.write('Dataset after filtering and NaN removal:', df_filter_droppedna)

               st.write('Correlation matrix:')
               df_corr2 = df_filter_droppedna.corr(method=corrtype)
               #df_pval = calculate_pvalues(df_filter_droppedna)

               if corrtype == 'pearson':
                df_corr = pearson_calculate_corr(df_filter_droppedna)

               elif corrtype == 'spearman':
                df_corr = spearman_calculate_corr(df_filter_droppedna)
           
               st.dataframe(df_corr)


               mask = np.zeros_like(df_corr2, dtype = float)
               mask[np.triu_indices_from(mask)] = True
               with sns.axes_style("white"):
                   fig, ax = plt.subplots(figsize=(10, 7))
                   ax = sns.heatmap(df_corr2, cmap="vlag", mask=mask, center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
               st.write(fig)

               st.write('p-values for correlation matrix:')

               if corrtype == 'pearson':
                df_pval = pearson_calculate_pvalues(df_filter_droppedna)

               elif corrtype == 'spearman':
                df_pval = spearman_calculate_pvalues(df_filter_droppedna)
           
               st.dataframe(df_pval)

           else:
               st.write('Click to start analysis')

       elif len(filter) <= 1:
           st.write('Select at least two variables for analysis.')



   ############ PAIR PLOTS ##################
       st.header('Part 2: Pair-plots')

       st.caption ("Pair-plots are scatter plots which help visualise the correlations between selected variables. There are 2 plots displayed for each pair of variables, one of which has a best-fit line. The diagonal for each variable is the histogram showing its distribution.")

       if len(filter) >= 2:
           if st.checkbox("Optional: enable sorting by categorical variable?", value=False):
               allvar = df.select_dtypes(include=['category','object','float64','int64']).columns
               filter2 = st.multiselect(
               'Select up to 10 variables of interest (include a categorical variable to stratify results if desired):', allvar)

               st.caption(':red[If an Error shows up, please ensure that you select a categorical variable TOGETHER with the variables selected in your correlation.]')
               st.write('  ')
               ### Sorting the list of selected variables, then filtering the dataset
               filter2 = sorted(filter2)
               df_filter2 = df.filter(items=filter2)

               df_filter2_droppedna = df_filter2.dropna(axis='rows', how = 'all')
               st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', df_filter2_droppedna.shape[0], 'of', df_filter2.shape[0])
               st.write('  ')
               catlist = df_filter2.select_dtypes(include=['category','object']).columns                ## extract list of categorical variables from original df
               sortby = st.selectbox(                                                           
                   'Select categorical variable to stratify your data by', catlist)

               filter3 = [f for f in filter2 if f not in catlist]

               st.write('Pair-plot of following variables:', str(filter3))
               pairplot = sns.pairplot(df_filter2_droppedna, vars = filter3, hue = sortby, dropna = True, diag_kind="kde")
               pairplot.map_lower(sns.regplot)
               # Save image of Pairplot before displaying on Streamlit - so it can be saved
               fn = 'pairplot.eps'
               plt.savefig(fn,format='eps')
               st.pyplot(pairplot)
           
               with open(fn, "rb") as img:
                    btn = st.download_button(
                    label="Download Pair Plot",
                    data=img,
                    file_name=fn,
                    mime="application/octet-stream"
                    )

           else:

               st.write('NAs are automatically removed here. Number of datapoints retained after removing NAs:', df_filter_droppedna.shape[0], 'of', df_filter.shape[0])
               pairplot = sns.pairplot(df_filter_droppedna, vars = filter, dropna = True, diag_kind="kde")
               pairplot.map_lower(sns.regplot)
               # Save image of Pairplot before displaying on Streamlit - so it can be saved
               fn = 'pairplot.eps'
               plt.savefig(fn,format='eps')
               st.pyplot(pairplot)
           
               with open(fn, "rb") as img:
                    btn = st.download_button(
                    label="Download Pair Plot",
                    data=img,
                    file_name=fn,
                    mime="application/octet-stream"
                    )

       elif len(filter) <= 1:
           st.write('You have not selected enough variables for analysis.')

   else: 
       st.write('Click to start Analysis!')






