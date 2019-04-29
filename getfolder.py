# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import calendar
from datetime import timedelta
import datetime
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
global estacao
import matplotlib.dates as mdates
from scipy.interpolate import spline
import matplotlib.patches as patches
import matplotlib as mpl
import os 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import mk_test

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')

def add_months(sourcedate,months):
    ''' Method to assist stack_cota with moth shifts'''
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

def read_cota(root,estacao_numero):
    estacao_teste_name = root + r'\cotas_T_' + str(int(estacao_numero)) +r'.txt'
    estacao_teste = pd.read_table(estacao_teste_name,encoding='iso-8859-1',header=9,sep=';',decimal=',',index_col=False,parse_dates=['Data'], date_parser=dateparse)
    return estacao_teste

def read_vazao(root, estacao_numero):
    estacao_teste_name = root + r'\vazoes_T_' + str(int(estacao_numero)) +'.txt'
    estacao_teste = pd.read_table(estacao_teste_name,encoding='iso-8859-1',header=9,sep=';',decimal=',',index_col=False,parse_dates=['Data'], date_parser=dateparse)
    return estacao_teste

def read_descarga(root,estacao_numero):
    estacao_teste_name = root + r'\ResumoDescarga_T_' + str(int(estacao_numero)) +'.txt'
    estacao_teste = pd.read_table(estacao_teste_name, header=6, encoding='iso-8859-1',sep=';',decimal=',',index_col=False)
    return estacao_teste

def stack_table(df,consistencia,field):
    ''' 
    Recives a structured stage dataframe (pd.DataFrame)
    COTA.txt of ANA (with set header set as 15),
    the consistency number wished:
    
    1 - Unconsisted
    2 - Consisted
    
    and statck it in a date/stage dataframe.
    
    in type: pd.DataFrame
    out type: pd.DataFrame
    
    '''
    if field == "Cota":
        printfield = "Stage"
    elif field == "Vazao":
        printfield = "Discharge"
    
    estacao = df["EstacaoCodigo"].values[0]
    
    dfc = df[df.NivelConsistencia == consistencia]
    if consistencia == 1:
        dfc = dfc[dfc.MediaDiaria == 1] # selects only non-consisted for dfc
    dfc.index = range(len(dfc)) #sets index in line for dfc
    dft = dfc.iloc[:,16:47] #selects only the gaging records in the hidroweb form 
    dft.index = dfc.Data
    dft.columns = range(1,32)
    df = dft
    
    print("--------Station name: %s---------" %(int(estacao)))
    if consistencia ==1:
        print("%s  registers unconsisted:" %(printfield), (len(df)))
    elif consistencia ==2:
        print("%s  registers consisted:" % (printfield), (len(df)))
    
    if len(df)!=0:
        vert = pd.DataFrame(index = pd.date_range(dft.index.min(),add_months(dft.index.max(),1)))
        dfcol = pd.DataFrame(df.stack())
        dfcol.columns =  [field]
        df2 = pd.DataFrame(dfcol.to_records())
        df2.columns = ['Data', 'Dia', field]
        df2.index = list(df2.Data+ pd.Series(map(lambda x: timedelta(days=float(x) - 1.), df2.Dia.tolist())))
        vert['Data'] = vert.index
        df2['Data'] = df2.index
        finalmente = pd.merge(vert,df2,on='Data')
        finalmente = vert.join(df2,lsuffix='_l',rsuffix='_r')
        finalmente['Mes'] = finalmente.index.month
        finalmente['Dia'] = finalmente.index.dayofyear
        finalmente['Ano'] = finalmente.index.year
        return finalmente
    else:
        return pd.Series(np.nan)
    
def merge_consistido(consistido,inconsistido):
    '''
    Merge consisted and unconsisted dataframes
    from stack_cota.
    
    in type: pd.DataFrame
    out type: pd.DataFrame
    
    '''
    if (any(consistido.notnull())) & (any(inconsistido.notnull())):
        frames = [consistido, inconsistido[inconsistido.Data_l>consistido.max().Data_l]]
        cotag = pd.concat(frames)
        cotag = cotag.sort_index()
        return cotag
    elif any(consistido.notnull()):
        cotag = consistido
        cotag = cotag.sort_index()
        return cotag
    elif any(inconsistido.notnull()):
        cotag = inconsistido
        cotag = cotag.sort_index()
        return cotag
    else:
        print('Error: Stage series is probably empty')
        exit()

def getfolder(root, estacoes_selecionadas): # Returns flow and discharge dictionaries for the given stations
    cota_dict = dict()
    vazao_dict = dict()
    descarga_dict = dict()
    for estacao in estacoes_selecionadas:
        dfc = read_cota(root,estacao)
        dfv = read_vazao(root,estacao)
        dfd = read_descarga(root,estacao)
        inconsistido_c = stack_table(dfc,1,'Cota')
        consistido_c = stack_table(dfc,2,'Cota')
        inconsistido_v = stack_table(dfv,1,'Vazao')
        consistido_v = stack_table(dfv,2,'Vazao')
        cota_dict[str(int(estacao))] = merge_consistido(consistido_c,inconsistido_c)
        vazao_dict[str(int(estacao))] = merge_consistido(consistido_v,inconsistido_v)
        #loc['1988-1-1':'2017-12-31']
        descarga_dict[str(int(estacao))] = dfd
    return cota_dict, vazao_dict, descarga_dict