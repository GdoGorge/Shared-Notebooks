# -*- coding: utf-8 -*-
"""
 

@author: G. GORGE
"""
import numpy as np
import pandas as pd
 
import pandas_profiling
 
from datetime import date


"""
PolicyAccountingPerMonth : calculation of main Accounting factors
"""

def PolicyAccountingPerMonth(data,periodrange=['2023-01','2023-02'],Stat=True):
      
      '''
      data : direct reading of Big Query Data Base (policy)
      periodrange : period of calculation
      '''

      # change in Value format
      ListvariableAccounting = ['Premium_HT_paid_to_Insurer','invoice_amount','Premium_HT']
      for variable in ListvariableAccounting:
              data[variable] = pd.to_numeric(data[variable], downcast="float")
      
      # Printing initial values
      if Stat:
            print('montants initiaux')
            dfStat = pd.DataFrame(data[ListvariableAccounting].sum(), columns=['Value'])

            dfStat.style.format(precision=0)
            display(dfStat)

      # change in Date format
      Listvariable = ['policy_period_effective_date','policy_period_expiration_date','policy_period_cancellation_date','policy_period_bound_date','bill_date']
      for variable in Listvariable:
                  data[variable] =  pd.to_datetime(data[variable]).dt.date 

      data['policy_period_expiration_date']=(data['policy_period_expiration_date']-pd.DateOffset(days=1)).dt.date  # we remove 1 day otherwise python calculate 366 days ;)
      
      # End of the policy to take into account cancellation
      data['real_policy_end'] = data['policy_period_expiration_date'].mask(data['policy_period_cancellation_date'].notna(),(data['policy_period_cancellation_date'] -pd.DateOffset(days=1)).dt.date)

      # Calculation of net Exposure
      data['is_real_cancellation'] = data['is_cancellation']
      data['is_real_cancellation'] = data['is_real_cancellation'].mask(data['policy_period_cancellation_date'].notna(), True)
      #display(data[['is_cancellation','policy_period_cancellation_date','is_real_cancellation']])

      # AccountingPeriod
      AccountingPeriod = pd.DataFrame(periodrange, columns=['accounting_period'])



      # selection of the accounting period
      AccountingPeriod['accounting_period_beginning'] = pd.to_datetime(AccountingPeriod['accounting_period'], format='%Y-%m', utc = True).dt.date # earned period start at this date (included)
      AccountingPeriod['accounting_period_end'] = (AccountingPeriod['accounting_period_beginning']+ pd.DateOffset(months=1)).dt.date  # earned period stop at this date (included)

      # Validation that AccountingPeriod includes the end of the most recent policy
      LatestPolicyEnd = data['real_policy_end'].max()
      LastestAccountingPeriod = AccountingPeriod['accounting_period_end'].max()

      if LastestAccountingPeriod <LatestPolicyEnd:
            print('Earned Policy will be wrong due to unsufficient Accounting Period')
            print(LatestPolicyEnd,LastestAccountingPeriod)
            return #we abord the function 

      # merge between the accounting tables and the accounting Periods
      df = data.merge(AccountingPeriod,how='cross')
      display(df.shape)
      # Duration of the policy in days
      df['policy_duration'] = ((df['real_policy_end'] - df['policy_period_effective_date']) / np.timedelta64(1, 'D')).apply(np.floor)

      # Calculation of earned Ratio (percentage that is earned in the specific period)
      df['Earned_period'] = ((df[['accounting_period_end','real_policy_end']].min(axis=1)
                              -df[['accounting_period_beginning','policy_period_effective_date']].max(axis=1))/ np.timedelta64(1, 'D')).apply(np.floor)

      # Gestion des cas de duration à 0. Voir si on garde cela
      df['policy_duration'] = df['policy_duration'].mask((df['policy_duration']==0),1)  # A partir du moment ou il y a une date de fin > date effective, on met un minimum un jour 
      df['policy_duration'] = df['policy_duration'].mask(df['policy_duration']<0,0) # eviter les durées négatives.
  
      # A partir du moment ou il y a une date de fin > date effective et que la date de début de la police est avant la fin de la période comptable , on met un minimum un jour 
      #df['Earned_period'] = df['Earned_period'].mask((df['Earned_period']==0)&(df['policy_duration']==1)&(df['accounting_period_end'] >=df['policy_period_effective_date']),1)  
      df['Earned_period'] = df['Earned_period'].mask((df['Earned_period']==0)&(df['policy_duration']==1),1)    
      df['Earned_period'] = df['Earned_period'].mask(df['Earned_period']<0,0) #  durées négatives indique pas d'exposition sur la période, on met 0.

      # Earned Ratio
      df['Earned_ratio'] = df['Earned_period']/ df['policy_duration']
      df['Earned_ratio'] = df['Earned_ratio'].mask(df['Earned_ratio'].isna(),0)

      # Written Ratio   
      df['written_ratio'] = 0
      #df['policy_period_bound_date']>df['accounting_period_beginning'])&(df['policy_period_bound_date']<df['accounting_period_end']
      df['written_ratio'] = df['written_ratio'].mask((df['policy_period_bound_date']>=df['accounting_period_beginning'])&(df['policy_period_bound_date']<df['accounting_period_end']),1) 

      #### Large validation to secure that we don't account too much
      ValidationWrittenPremium = df.groupby(by=['policy_number','invoice_number'], as_index=False)[['written_ratio','Earned_ratio']].sum().rename(columns={'written_ratio':'SumWritten_ratio','Earned_ratio':'SumEarned_ratio'})
      df = df.merge(ValidationWrittenPremium,on=['policy_number','invoice_number'])
      display(df.shape)
      df['written_ratio'] = df['written_ratio'] / df['SumWritten_ratio']
       
      df['Earned_ratio'] = df['Earned_ratio'] / df['SumEarned_ratio']
      # Traitement des annulations "brutal"
      df['Earned_ratio'] = df['Earned_ratio'].mask((df['written_ratio'] == 1)&(df['SumEarned_ratio']==0),1)

      # Validation through the calculation of Earned Premium
      df['Earned_Premium_HT_paid_to_Insurer'] = df['Premium_HT_paid_to_Insurer']* df['Earned_ratio']
      df['Written_Premium_HT_paid_to_Insurer'] = df['Premium_HT_paid_to_Insurer']* df['written_ratio']


      # Calculation of Number of Policies
      PolicyNbr = df.groupby(by=['policy_period_id'], as_index=False)[['written_ratio','Earned_ratio']].sum().rename(columns={'written_ratio':'Total_Written_Exposure_perPolicy','Earned_ratio':'Total_Earned_Exposure_perPolicy'})
      
      df = df.merge(PolicyNbr,how='left',on='policy_period_id')
      df['Written_Exposure'] = df['written_ratio'] / df['Total_Written_Exposure_perPolicy']
      df['Earned_Exposure'] = df['Earned_ratio'] / df['Total_Earned_Exposure_perPolicy']

      # We remove policy with payment accident
      df['is_cancellation'] = df['is_cancellation'].mask(df['is_cancellation'].isna(),False)
      df['Net_written_Exposure'] = df['Written_Exposure'].mask(df['is_real_cancellation'],0) 

      display(df.shape)

      # print stat
      if Stat:
            dfStat = pd.DataFrame(df[['Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer','Written_Exposure','Net_written_Exposure','Earned_Exposure' ]].sum(), columns=['Value'])
            dfStat.style.format(precision=0)      
            display(dfStat)

      
      # suppress all the lines that are useless
      df = df[(~((df['Earned_period'] == 0) | (df['policy_duration']==0))|(~(df['written_ratio']==0)))]

      # Date of the day
      df['DayDate'] = date.today()
      display(df.shape)

      # print stat
      if Stat:       
            dfStat = pd.DataFrame(df[['accounting_period','Written_Exposure','Net_written_Exposure','Earned_Exposure','Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer',]].groupby(by=['accounting_period'],as_index=False).sum())
            dfStat.style.format(precision=0)      
            display(dfStat)

      return df

"""
UnEarnedPremiumCalculation : Put in format NOT USED ANYMORE
"""

def UnEarnedPremiumCalculation(Accounting,accountingMonth="2023-06",Stat=False,variableStat = ['Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer','Unearned_Premium_HT_paid_to_Insurer','Earned_Exposure','Written_Exposure','Net_written_Exposure',]
                               ,varSegmentation=['ReinsurerSegment']):
    ''' 
    Tout cela pour cela... je suis obligé de faire cette monstruosité pour gérer la prime non acquise
    '''
    df = Accounting.copy()


    df['reporting_period']  = accountingMonth

    df['reporting_period_beginning'] = pd.to_datetime(df['reporting_period'], format='%Y-%m', utc = True).dt.date 
    df['reporting_period_end'] = (df['reporting_period_beginning']+ pd.DateOffset(months=1)).dt.date  # reporting period stop at this date (included) ON peut surement accélerer cette partie sans passer par PANDAs !!!

    df['Unearned_Ratio'] = 0
    df['Unearned_Ratio'] = df['Unearned_Ratio'].mask(((df['accounting_period_beginning']>df['reporting_period_beginning'])&(df['policy_period_bound_date']<df['reporting_period_end'])),1) 
    # il faut aussi vérifier que la police ait été effectivement souscrite avant la période
    

    df['reporting_Ratio'] = 0
    df['reporting_Ratio'] = df['reporting_Ratio'].mask((df['accounting_period_beginning']==df['reporting_period_beginning']),1) 

    df['Earned_Premium_HT_paid_to_Insurer'] = df['Premium_HT_paid_to_Insurer']* df['Earned_ratio'] * df['reporting_Ratio']
    df['Written_Premium_HT_paid_to_Insurer'] = df['Premium_HT_paid_to_Insurer']* df['written_ratio'] * df['reporting_Ratio']
    df['Unearned_Premium_HT_paid_to_Insurer'] = df['Premium_HT_paid_to_Insurer']* df['Earned_ratio']* df['Unearned_Ratio']
    df['Earned_Exposure'] =  df['Earned_ratio'] * df['reporting_Ratio'] 

    # Calcul du nombre de polices - un peu compliqué mais cohérent
    df['Written_Exposure'] =  df['written_ratio'] * df['reporting_Ratio'].mask(df['Premium_HT_paid_to_Insurer'] ==0,0)
    Total_Written_Exposure_perPolicy = df[['policy_period_id','Written_Exposure','Written_Premium_HT_paid_to_Insurer']].groupby(by='policy_period_id', as_index=False).agg(
    Total_Written_Exposure_perPolicy=pd.NamedAgg(column="Written_Exposure", aggfunc="sum"),
    Total_Written_Premium_HT_paid_to_Insurer=pd.NamedAgg(column="Written_Premium_HT_paid_to_Insurer", aggfunc="sum")) 
    df = df.merge(Total_Written_Exposure_perPolicy,how='left',on='policy_period_id')
    df['Written_Exposure'] = df['Written_Exposure'] / df['Total_Written_Exposure_perPolicy']

    # On calcule le nombre de polices Earned en fonction de la prime acquise
    df['Earned_Exposure'] = 0
    df['Total_Written_Premium_HT_paid_to_Insurer'] =df['Total_Written_Premium_HT_paid_to_Insurer'].mask((df['Total_Written_Premium_HT_paid_to_Insurer'].isna()),1)
    df['Total_Written_Premium_HT_paid_to_Insurer'] = df['Total_Written_Premium_HT_paid_to_Insurer'].mask((df['Total_Written_Premium_HT_paid_to_Insurer']==0),1)
    df['Total_Written_Premium_HT_paid_to_Insurer'] = df[['Total_Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer']].max()
    df['Earned_Exposure'] = df['Earned_Premium_HT_paid_to_Insurer'].where((df['Earned_Premium_HT_paid_to_Insurer'].notna()),0) / df['Total_Written_Premium_HT_paid_to_Insurer'] 
    #display(df[['Earned_Exposure','Earned_Premium_HT_paid_to_Insurer','Premium_HT_paid_to_Insurer','Total_Written_Premium_HT_paid_to_Insurer']][(df['Earned_Exposure'].isna())])  
 

    # We remove policy with payment accident
    df['is_cancellation'] = df['is_cancellation'].mask(df['is_cancellation'].isna(),False)
    df['Net_written_Exposure'] = df['Written_Exposure'].mask(df['is_cancellation'],0) 

 
    #df.drop(columns = ['Total_Written_Exposure_perPolicy','Total_Earned_Exposure'])

    #display(df.head(2))

    #Main Stat
    vartoKeep = ['reporting_period'] + variableStat + varSegmentation
    varGroupby = ['reporting_period'] + varSegmentation
    dfStat = pd.DataFrame(df[vartoKeep].groupby(by=varGroupby, as_index=False).sum())
    if Stat:

            display(dfStat)
            # display(    df.groupby(by=['is_cancellation','is_down_payment'])[variable].sum())
            # nombre de polices display(df['policy_number'].value_counts().count())
 

    return df, dfStat



"""
ClaimsAccountingPerMonth : calculation of main Accounting factors
"""
 
def ClaimsAccountingPerMonth(data,periodrange=['2023-01','2023-02'],Stat=True,variableStat = ['NbrClaims','TotalClaimsCost','TotalClaimsffReserve','ClosedFile','TotalClosedFileCost','BIFile','TotalBIClaimsCost','TotalBIClaimsffReserve','WindScreenFile','TotalWindScreenClaimsCost'   ]
                             ,varSegmentation=['ReinsurerSegment']
                             ):
      
      '''
      data : direct reading of Big Query Data Base (claims) 
      periodrange : period of calculation
      '''

      df=data.copy()
      # Change in amount format
      ListvariableAccounting = ['paid','recovered','reserved','cost',]
      for var in ListvariableAccounting:
              df[var] = pd.to_numeric(df[var], downcast="float")

      # Printing initial values
      if Stat:
            print('montants initiaux')
            display(pd.DataFrame(df[ListvariableAccounting].sum(), columns=['Value']))

      # change in Date format
      ListvariableClaimsDate = ['claim_date',]
      for varClaims in ListvariableClaimsDate:
            df[varClaims] =  pd.to_datetime(df[varClaims]).dt.date 

 
      # AccountingPeriod
      AccountingPeriod = pd.DataFrame(periodrange, columns=['accounting_period'])

      # selection of the accounting period
      AccountingPeriod['accounting_period_beginning'] = pd.to_datetime(AccountingPeriod['accounting_period'], format='%Y-%m', utc = True).dt.date # earned period start at this date (included)
      AccountingPeriod['accounting_period_end'] = (AccountingPeriod['accounting_period_beginning']+ pd.DateOffset(months=1)).dt.date  # earned period stop at this date (included)

      # merge between the accounting tables and the accounting Periods
      df = df.merge(AccountingPeriod,how='cross')

      # Claims calculation


      # Allocated to the accounting period
      df['NbrClaims'] = 0
      df['NbrClaims'] = df['NbrClaims'].mask(((df['claim_date']>=df['accounting_period_beginning'])&(df['claim_date']<df['accounting_period_end'])),1) 
      df['TotalClaimsCost'] = df['NbrClaims'] * df['cost']
      df['TotalClaimsffReserve'] = df['NbrClaims'] * df['reserved']
      
      # Close file
      df['ClosedFile'] = 0
      df['ClosedFile'] = df['ClosedFile'].mask((df['phase']=='CLOTURE'),1) *df['NbrClaims']
      df['TotalClosedFileCost'] = df['ClosedFile'] * df['cost']
      
      # BI file
      df['BIFile'] = 0
      df['BIFile'] = df['BIFile'].mask((df['nature']!='MATÉRIEL'),1) *df['NbrClaims']
      df['TotalBIClaimsCost'] = df['BIFile'] * df['cost']
      df['TotalBIClaimsffReserve'] = df['BIFile'] * df['reserved']


      # Windscreen
      df['WindScreenFile'] = 0
      df['WindScreenFile'] = df['WindScreenFile'].mask((df['accident_type'].isin(['BRIS DE GLACE REMPLACÉ','BRIS DE GLACE RÉPARÉ'])),1) *df['NbrClaims']
      df['TotalWindScreenClaimsCost'] = df['WindScreenFile'] * df['cost']
      df['TotalWindScreenClaimsffReserve'] = df['WindScreenFile'] * df['reserved']
      	
      #TableVerification = pd.DataFrame(df[['claim_id','cost','NbrClaims']].groupby(by=['claim_id'],as_index=False).sum())
      #display(TableVerification)
      #display(df[['claim_id','accounting_period_beginning','accounting_period_end','claim_date','cost','NbrClaims']][(df['claim_id']=='SONK2300021')])
      #Main Stat
      #Main Stat
      vartoKeep = ['accounting_period'] + variableStat + varSegmentation
      varGroupby = ['accounting_period'] + varSegmentation
      dfStat = pd.DataFrame(df[vartoKeep].groupby(by=varGroupby, as_index=False).sum())
 
      if Stat:
                  display(dfStat)
                  display(dfStat.TotalClaimsCost.sum())
                  # display(    df.groupby(by=['is_cancellation','is_down_payment'])[variable].sum())
                  # nombre de polices display(df['policy_number'].value_counts().count())
      

      return df, dfStat


"""
AccountingValidation : Validation of the consistency of the results
"""
 

def AccountingValidation(Accounting,dfAccounting,df,policynumber=1,variableAccounting = ['policy_number','invoice_number','invoice_id','Premium_HT_paid_to_Insurer','invoice_amount','Premium_HT','policy_period_effective_date','real_policy_end','policy_period_expiration_date','policy_period_cancellation_date','policy_period_bound_date','bill_date','accounting_period','Earned_period','written_ratio','update_time']):
    AccountingPerPolicy = Accounting[['policy_number','policy_period_id','Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer','Premium_HT_paid_to_Insurer','Earned_period']].groupby(by=['policy_number','policy_period_id'] , as_index=False).sum()

    # policies with written premium
    PolicywithProblem = AccountingPerPolicy[(AccountingPerPolicy['Earned_Premium_HT_paid_to_Insurer']>AccountingPerPolicy['Written_Premium_HT_paid_to_Insurer']+0.1)]
    
    #[(ComptaPrime['policy_number']=='ONK0011222-W')]

    print(str(PolicywithProblem.shape[0])+' Potential policies with issue')
    if PolicywithProblem.shape[0] > 0:
        PolicywithProblem = PolicywithProblem.iloc[0:policynumber,] 
        display(PolicywithProblem)
        AccountingPolicywithProblem=PolicywithProblem[['policy_number']].merge(dfAccounting[variableAccounting],on='policy_number')
        display(AccountingPolicywithProblem)
        dfPolicywithProblem = PolicywithProblem[['policy_number']].merge(df[['policy_number','submission_created_at','total_premium_rpt', 'bind_date','period_start',  'policy_period_start',	'policy_period_end']], left_on='policy_number', right_on = 'policy_number')
        display(dfPolicywithProblem)

    # Verification qu'on ne perd pas de prime entre prime émise et prime acquise au niveau d'une quittance
 
    InvoiceWithPb = Accounting.groupby(by=['policy_number','invoice_number'], as_index=False)[['Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer']].sum()
    InvoiceWithPb = InvoiceWithPb[(InvoiceWithPb['Written_Premium_HT_paid_to_Insurer']- InvoiceWithPb['Earned_Premium_HT_paid_to_Insurer'] >0.01)|(InvoiceWithPb['Earned_Premium_HT_paid_to_Insurer']- InvoiceWithPb['Written_Premium_HT_paid_to_Insurer'] >0.01)]
    print(str(InvoiceWithPb.shape[0])+' Potential quittances with issue')
    if InvoiceWithPb.shape[0] > 0:
        InvoiceWithPb = InvoiceWithPb.iloc[0:policynumber,] 
        display(InvoiceWithPb)
        AccountingPolicywithProblem=InvoiceWithPb[['invoice_number']].merge(dfAccounting[variableAccounting],on='invoice_number')
        display(AccountingPolicywithProblem)
        dfPolicywithProblem = InvoiceWithPb[['policy_number']].merge(df[['policy_number','submission_created_at','total_premium_rpt', 'bind_date','period_start',  'policy_period_start',	'policy_period_end']], left_on='policy_number', right_on = 'policy_number')
        display(dfPolicywithProblem)

"""
AccountingReinsurer : Calculation of reinsurer database
"""

def AccountingReinsurer(Accounting ,df, period,Stat=False,varSegmentation=['ReinsurerSegment'] ):
      '''
      # period is the date used for calculation of Unearned Premium : all the periods after it are not retained but used to calculate unearned Premium

      '''
      VarMerge = ['policy_number']+ varSegmentation 
      VarGroupingwithUnearnedPremium = ['accounting_period']+varSegmentation
      VarGrouping = ['accounting_period','DayDate']+varSegmentation
        
      VarStat = varSegmentation+['accounting_period','DayDate','Written_Exposure','Net_written_Exposure','Earned_Exposure','Written_Premium_HT_paid_to_Insurer','Earned_Premium_HT_paid_to_Insurer',]
      print(VarStat)
      print(VarGrouping)
      # merge with dfPolicy
      Accounting = Accounting.merge(df[VarMerge],left_on='policy_number', right_on = 'policy_number')

      # Statistic per period and varSegmentation
      dfStatComplete=Accounting[VarStat].groupby(by=VarGrouping,as_index=False).sum()
      #iteration on each reporting period
   
      # Calculation of Unearned Premium
      dfStatComplete['Unearned_Premium_HT_paid_to_Insurer'] = dfStatComplete['Earned_Premium_HT_paid_to_Insurer'].mask((dfStatComplete['accounting_period']<=period),0)
      UnearnedPremium = dfStatComplete[varSegmentation+['Unearned_Premium_HT_paid_to_Insurer']].groupby(by=varSegmentation,as_index=False).sum()
      UnearnedPremium['accounting_period'] = period
      dfStatComplete = dfStatComplete[(dfStatComplete['accounting_period']<=period)].drop(columns=['Unearned_Premium_HT_paid_to_Insurer'])
      dfStatCompletewithUnearnedPremium = dfStatComplete.merge(UnearnedPremium,on = VarGroupingwithUnearnedPremium,how='left')
      if Stat:
            display(dfStatCompletewithUnearnedPremium)
      return dfStatCompletewithUnearnedPremium