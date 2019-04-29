
# coding: utf-8

# Análise Vazões

# In[1]:




# In[2]:


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
import getfolder

# In[3]:


plt.rc('text', usetex=False)
plt.rc('font', family='Arial')


# In[4]:


root = r'D:\OneDrive\Programacao\mygit\2.python_hidroweb_data_analyzer\exemplo_66090000'

estacoes_selecionadas =[
66120000]

# Problemas
#66825000 - Ladário
#66260001 - Cuiabá

estacoes_selecionadas_str = []
for i in estacoes_selecionadas:
    estacoes_selecionadas_str.append(str(int(i)))

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')


def var_erro_anual(cotg,nr_p,estacao,figuras,nome,Pmax): ###NO USE SO FAR###
    '''
    Receives the date/stage dataframe (cotg),
    the percentile and errors dataframe (nr_p),
    the station number and name for the title (estacao, nome),
    the address to save the output figure (figuras),
    and the maximum values of years to the past
    used for percentile calculations (Pmax).
    
    Built the plot that represents the errors distributions
    with different P values (1 to Pmax).
    
    input type: (pd.DataFrame, pd.DataFrame, int, str, str, int)
    output type: matplotlib figure and pdf file.
    
    '''
    
    rc('font',family='Sans-Serif',size=10) 
    dados_anuais = pd.pivot_table(nr_p,index=[0], columns='P')['Erro_med_1']
    dados_anuais2 = dados_anuais.dropna(subset = [range(1,Pmax)])
    left, width = 0.125, .7750
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.04
    lsup= (cotg.index.year.max()-4)
    linf= (cotg.index.year.min()+Pmax-1)
    nlinhas = Pmax *(lsup - linf)

    plt.figure(figsize(9,4.5))
    dados_anuais.index+=1
    sns.heatmap(dados_anuais[dados_anuais.notnull()].T[::-1],cmap="inferno",
                xticklabels=5,cbar_kws={"orientation": "horizontal"})
    xlabel('Year')
    ylabel('P (years)')
    plt.yticks(rotation=0)
    plt.text(-4,-5,u'Avg.\nError\n(cm)',ha='center',va='center',rotation='vertical')

    nullfmt = ticker.NullFormatter()
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    axHistx = plt.axes(rect_histx) # x histogram
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.scatter(dados_anuais.index.tolist(),dados_anuais.T.idxmin().tolist(),
                    edgecolor=None,marker='x')
    axHistx.plot(dados_anuais.index.tolist(),dados_anuais.T.idxmin().tolist(),lw=0.3)
    
    axHistx.set_xlim(linf-0.5,lsup-.5)
    axHistx.set_ylim(0,20)#Pmax)
    axHistx.set_xticks(range(linf,lsup,2))
    axHistx.set_title(nome+'-'+str(estacao)+'-'+str(cotg.index.year.min())+
                      ':'+str(cotg.index.year.max())+
                      u'\n Error as function of P for 1 year prediction.')
    
    axHistx.set_ylabel( 'P (years)')

    left, width = 0.1, 0.81
    bottom, height = 0.34, 0.58
    bottom_h = left_h = left+width+0.02

    rect_histy = [left_h, bottom, 0.25, height]
    axHisty = plt.axes(rect_histy)

    axHisty.scatter(dados_anuais2.mean(skipna=0).tolist(),dados_anuais2.T.index.tolist()[::],
                    edgecolor=None,marker='x',label='Avg.',c='r')
    dados_anuais2.boxplot(vert=0,showmeans=0,ax=axHisty)

    startx, endx = axHisty.get_xlim()
    starty, endy = axHisty.get_ylim()

    axHisty.set_xlim(0,210)
    axHisty.set_yticks(range(int(starty)+1,int(endy)+1,1))
    axHisty.set_xlabel(u'Avg. Error Distribution (cm)')
    axHisty.legend(bbox_to_anchor=(.02, 1.1), loc=2, borderaxespad=0.)
    axHisty.set_ylim(0,21)#Pmax)
    savefig(figuras+str(estacao)+'_VariacoesAnuais_P_small_sides_1_eng.pdf',
            bbox_inches='tight')


def plot_variacao_anual(cotg,estacao,figuras,nome): ### NO USE SO FAR###
    # Retirando percentils calculados sem dados suficientes (dados_suficientes)
    dados_suficientes = 347 # 95%
    caux1 = pd.pivot_table(cotg,index=['Dia','Mes'], columns='Ano')['Cota']
    caux2 = caux1.drop_duplicates()
    caux3 = caux2.notnull().sum() >= dados_suficientes
    caux4 = caux1.quantile([.1]).T
    caux5 = pd.DataFrame(caux3)[0]* caux4[caux4.columns[0]]
    caux5[caux5==0] = np.nan
    
    percentil = pd.pivot_table(cotg,index=['Dia','Mes'], columns='Ano')['Cota'].quantile([.10]).T
    percentil = caux5
    #Ca
    dados_anuais2 = pd.pivot_table(cotg,index=['Ano'],columns='Dia')['Cota']
    meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez',]
    dados_anuais3 = dados_anuais2.T
    dados_anuais3.index.name = None
    left, width = 0.125, .7750
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.04

    lsup= (cotg.index.year.max())
    linf= (cotg.index.year.min())
    nlinhas = Pmax *(lsup - linf)

    plt.title(u'Cota (cm) - '+estacao+' - '+ str(linf) + ':' + str(lsup))
    sns.heatmap(dados_anuais3,cmap="viridis",vmin=cotg.Cota.min(), vmax=cotg.Cota.max(), cbar_kws={"orientation": "horizontal"})
    plt.yticks(np.linspace(15,380,13),meses[::-1],rotation=-45)
    #plt.xticks(np.linspace(0,365,13),['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez',])
    #xlabel(u'Mês')
    plt.ylim(0,366)


    nullfmt = ticker.NullFormatter()
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    axHistx = plt.axes(rect_histx) # x histogram
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.plot(percentil.index.tolist()[::],percentil.tolist()[::],)
    axHistx.set_xlim(linf-0.5,lsup+.5)
    axHistx.set_title(nome+' - '+estacao+'-'+str(cotg.index.year.min())+'-'+str(cotg.index.year.max())+u'\n Percentil 10 (cm)')
    axHistx.scatter(percentil.index.tolist()[::],percentil.tolist()[::],marker='x',s=60)
    axHistx.set_ylabel( 'NR - Percentil 10 (cm)')
    #savefig(figuras+estacao+'_VariacoesAnuais.png',bbox_inches='tight')
    plt.show()
    #plt.close()


def plot_descarga(estacao_numero):
    plt.scatter(estacao_teste.Cota, estacao_teste.Vazao)
    plt.show()

def set_cota_vazao(df_cota,df_vazao):
    merged = df_vazao.merge(df_cota,left_index=True,right_index=True)
    return merged

def plot_chave_dispersao(df_cota_vazao, df_resumo_descarga):
    plt.scatter(df_cota_vazao.Vazao, df_cota_vazao.Cota,label = 'observados',s=1)
    plt.scatter(df_resumo_descarga.Vazao, df_resumo_descarga.Cota, label='curvas cota-chave', s=8)
    plt.legend()
    plt.grid()
    plt.show()

def plot_daily_flow_imshow(df,nome,output,field,dados_suficientes = 347,cmap="viridis",tipo='boxplot'):
    cotg = df[:-1]
    
    max_flow = cotg["Vazao"].max()
    min_flow = cotg["Vazao"].min()
    mean_flow = cotg["Vazao"].mean()
    
    print("Station %s - max flow: %.2f cms" % (estacao,max_flow))
    print("Station %s - min flow: %.2f cms" % (estacao,min_flow,))
    print("Station %s - mean flow: %.2f cms" % (estacao,mean_flow))
    
    Pmax = 20
    fig = plt.figure(figsize=(10,6))
    detalhes = dict()
    detalhes['Vazao'] = {'nome':u'Vazão'+r' ($m^{3}/s$)', 'unidade':r'$m^{3}/s$'}
    detalhes['Cota'] = {'nome':'Cota', 'unidade':'cm'}
    
    if tipo == 'boxplot':
        
        caux1 = pd.pivot_table(cotg,index=['Dia','Mes'], columns='Ano')[field]
        caux2 = caux1.drop_duplicates()
        caux3 = caux2.notnull().sum() >= dados_suficientes
        caux1.T[~caux3] = np.nan
        caux1.columns = pd.to_datetime(caux1.columns, format='%Y')
        resumo = caux1.T.asfreq(freq='AS').T

    if tipo == 'percentil':
        caux1 = pd.pivot_table(cotg,index=['Dia','Mes'], columns='Ano')[field]
        caux2 = caux1.drop_duplicates()
        caux3 = caux2.notnull().sum() >= dados_suficientes
        caux4 = caux1.quantile([.1]).T
        caux5 = pd.DataFrame(caux3)[0]* caux4[caux4.columns[0]]
        caux5[caux5==0] = np.nan
        percentil = pd.pivot_table(cotg,index=['Dia','Mes'], columns='Ano')[field].quantile([.10]).T
        percentil = caux5
    
    dados_anuais2 = pd.pivot_table(cotg,index=['Ano'],columns='Dia')[field]
    meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez',]
    dados_anuais3 = dados_anuais2.T
    dados_anuais3.index.name = None
    left, width = 0.125, .7750
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.04

    lsup= (cotg.index.year.max())
    linf= (cotg.index.year.min())
    nlinhas = Pmax *(lsup - linf)
    
    plt.title(detalhes[field]['nome'] + ' - '+estacao+' - '+ str(linf) + ':' + str(lsup))
    fp = sns.heatmap(dados_anuais3,cmap=cmap,vmin=cotg[field].min(), vmax=cotg[field].max(), cbar_kws={"orientation": "horizontal"})
    plt.yticks(np.linspace(15,380,13),meses[::-1],rotation=-45)
    #plt.xticks(np.linspace(0,365,13),['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez',])
    #xlabel(u'Mês')
    plt.ylim(0,366)
    nullfmt = ticker.NullFormatter()
    """
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    axHistx = plt.axes(rect_histx) # x histogram
    
    if tipo == 'boxplot':
        #sns.boxplot(resumo,ax = axHistx)
        resumo.boxplot(ax = axHistx,autorange=True,showfliers=True)
        #axHistx.set_xlim(fp.get_xlim()[0]+.5,fp.get_xlim()[1]+.5)
        print(fp.get_xlim()[0])
        print(fp.get_xlim()[1])
        axHistx.set_title(estacao+'-'+str(cotg.index.year.min())+'-'+str(cotg.index.year.max())+ " - " + u'{} ({})'.format(detalhes[field]['nome'], ' '+ detalhes[field]['unidade']))
        axHistx.set_ylabel( u'{} ({})'.format(detalhes[field]['nome'], detalhes[field]['unidade']) )
    
    if tipo == 'percentil':
        axHistx.plot(percentil.index.tolist()[::],percentil.tolist()[::],)
        axHistx.scatter(percentil.index.tolist()[::],percentil.tolist()[::],marker='x',s=60)
        axHistx.set_xlim(linf-0.5,lsup+.5)
        axHistx.set_title(estacao+'-'+str(cotg.index.year.min())+'-'+str(cotg.index.year.max())+ u'{} ({})'.format(detalhes[field]['nome'], detalhes[field]['unidade']))
        axHistx.set_ylabel( 'NR - Percentil 10 (cm)')
    
    #axHistx.xaxis.set_major_formatter(nullfmt) 
    """
    plt.savefig(nome+'.png',bbox_inches='tight',dpi=300)
    plt.show()
              
def print_flow_monthly_average(df,estacao,dic_names_rivers):
    
    dict_stations = dic_names_rivers
    
    final_array = np.zeros((12,1))
    series_monthly_mean = df["Vazao"].resample('M').mean()
    
    yticks = np.linspace(0,200,5)
    inicio = df.iloc[0]["Ano"]
    fim = df.iloc[-1]["Ano"]
    
    #Acha as séries mensais 
    
    #series_monthly_max = df["Vazao"].resample('M').max()
    #series_monthly_min = df["Vazao"].resample('M').min()
    series_monthly_min = df["Vazao"].resample('M').mean()
    
    
    date_array = np.arange(1,13,1)
    for i in range (1,13,1):

        series_mean = series_monthly_mean[series_monthly_mean.index.month.isin([i])].mean()
        #series_max = series_monthly_max[series_monthly_max.index.month.isin([i])].mean()
        #series_min = series_monthly_min[series_monthly_min.index.month.isin([i])].mean()
        
        final_array[i-1,0] = series_mean
        #final_array[i-1,1] = series_max
        #final_array[i-1,2] = series_min
        
    #max_flow = df["Vazao"].max()
    #min_flow = df["Vazao"].min()
    media = df["Vazao"].mean()
                  
    xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    #media_x = final_array[:,0].mean()
    media_y = np.zeros(12)+media
    media_x = np.arange(1,13,1)

    fig, ax = plt.subplots(figsize=(8,8))
    
    x_smooth = np.linspace(1,12,300)
    y_smooth = spline(date_array, final_array[:,0], x_smooth)
    
    ax.fill_between(x_smooth, y_smooth, cmap = "Blues", color = 'darkblue',alpha = 0.8)
    ax.set_ylabel('Vazao'+" ("+ r'$m^{3}/s$' + ")", fontsize=20)
    ax.set_xlabel('%s - %s'%(inicio,fim), fontsize=20)
    ax.set_yticks(yticks)
    
    ax.plot(media_x, media_y,color = "orange")
    #ax.scatter(date_array, final_array[:,0], color = "red")
    ax.annotate('Qm = %.0f ' %media_y[0] + r'$m^{3}/s$' , xy=(6, media_y[0]), xytext=(6, media_y[0]+10), fontsize=27)
    ax.annotate('%s'% estacao , xy=(0.41, 0.89), xytext=(0.41, 0.89), xycoords='figure fraction', fontsize=25, annotation_clip=False)
   
    ax.set_xticklabels(xticks)
    ax.set_xticks(date_array)
    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.grid(which='both', axis='y')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    
    #plt.tight_layout()
    #plt.ylim(0,250)

    ax.margins(x=0)
    plt.title(dic_names_rivers[estacao][0] + " - " + dic_names_rivers[estacao][1], y=1.07, fontsize = 30)
    filetarget = (os.path.join(root, "\%s_plot.png"% estacao))
    plt.savefig(filetarget,dpi = 300)
    plt.show()
    
def print_stagedischarge(estacoes_selecionadas_str,vazao_dict,cota_dict,descarga_dict):
    for i in estacoes_selecionadas_str: # prints a stage-discharge curve
        fig = plt.figure(figsize=(20,10))
        vazaog = vazao_dict[i].dropna()
        cotag = cota_dict[i].dropna()
        resumog = descarga_dict[i]
        merged = set_cota_vazao(cotag,vazaog)
        plt.title("Curva Cota-chave - Estacao"+ i)
        plot_chave_dispersao(merged, resumog)
        fig = plt.figure(figsize=(20,10))
        ax = plt.subplot(1,2,1)
        ax.scatter(cotag.index,cotag.Cota,s=1, label = "Cotas")
        ax.scatter(vazaog.index,vazaog.Vazao,s=1, label = "Vazões")
        ax.set_title("Vazoes vs Cotas")
        ax.set_ylabel("Vazoes" + r'$m^{3}/s$')
        ax.set_xlabel("Anos")
        ax.grid()
        ax.legend()
        ax2 = plt.subplot(1,2,2)
        mg = ax2.scatter(merged.Vazao, merged.Cota,c=merged.Ano_x,s=1,cmap='viridis', edgecolor=None, label = "Vazoes observadas")
        fig.colorbar(mg)
        ax2.scatter(x=resumog.Vazao, y=resumog.Cota,c='r',edgecolor=None,s=1, label = "Vazoes medidas")
        ax2.set_title("Curva Cota-chave")
        ax2.set_ylabel("Vazoes" + r'$m^{3}/s$')
        ax2.set_xlabel("Cotas(cm)")
        ax2.grid()
        ax2.legend()
        plt.savefig("stagedischarge_{0}.png".format(i))
        plt.show()
        
def print_linearseries(descarga_dict, dict_stations, resampler_key):
    for k in descarga_dict:
        station_name = dict_stations[k][0]
        river_name = dict_stations[k][1]
        fig = plt.figure(figsize = (10,5))
        df = vazao_dict[k]
        df.set_index("Data_r", inplace = True)
        df.index = pd.to_datetime(df.index)
        series = df["Vazao"] 
        series = series.resample(resampler_key).mean()
        series.plot()
        plt.xlabel("Data", fontsize=18)
        #plt.xlim('1988-1-1', '2017-12-31')
        #xticks = pd.date_range('1988-1-1', '2017-12-31', freq='4Y')
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.ylabel(r"Vazao $m^{3}/s$", fontsize=18)
        #plt.title("Estacao %s - %s - %s"%(k,station_name,river_name), fontsize=24)
        plt.grid(which = 'major')
        plt.tight_layout()
        plt.savefig(k+"meanflow.png", dpi = 300)
        plt.show()

def print_correlogram(vazao_dict, dict_stations, savefig = False):
    corrilations = {}
    for k in descarga_dict:
        station_name = dict_stations[k][0]
        river_name = dict_stations[k][1]
        #fig = plt.figure(figsize = (7.5,3))
        df = vazao_dict[k]
        df.set_index("Data_l", inplace = True)
        df.index = pd.to_datetime(df.index)
        series = df["Vazao"] 
        series = series.resample('M').mean()
    
        datacorr = series.values
        datacorr = datacorr[~np.isnan(datacorr)]
        j = sm.tsa.stattools.acf(datacorr, unbiased=False, nlags=10, qstat=False, fft=None, alpha=0.05, 
                                          missing='raise')
        
        if savefig == True:
            plot_acf(datacorr, ax=None, lags=10, alpha=0.05, use_vlines=True, unbiased=False, 
         fft=False, title='Autocorrelation - %s - %s'%(station_name, river_name), zero=True, vlines_kwargs=None)
            plt.savefig("autocorr_"+"k.png")
        
        corrilations[k]  = j[0]
    return corrilations

def call_mk_test(vazao_dict, dict_stations,corrilations):
    mk_values = {} 
    for k in descarga_dict:  
        
        acorrilation = corrilations[k][1]
        
        df = vazao_dict[k]
        df.set_index("Data_r", inplace = True)
        df.index = pd.to_datetime(df.index)
        series = df["Vazao"] 
        series = series.resample('M').mean()
        datacorr = series.values
        datacorr = datacorr[~np.isnan(datacorr)]
        
        datacorr_lag = []
        for o in range(len(datacorr)-1):
            datacorr_lag.append(datacorr[o+1] - acorrilation*datacorr[o])
        datacorr_lag = np.array(datacorr_lag)
        mk_values[k] = mk_test.mk_test(datacorr_lag)
    return mk_values

########################################Commands###############################

dict_stations_rivers = {
            "66120000": ["Porto Conceicao","Rio Paraguai"],
                    }    
cota_dict, vazao_dict, descarga_dict = getfolder.getfolder(root, estacoes_selecionadas)

for i in range(len(estacoes_selecionadas)):
    estacao = str(int(estacoes_selecionadas[i]))
    cotg = vazao_dict[estacao]
    
    print_flow_monthly_average(cotg,estacao, dict_stations_rivers)
    
    plot_daily_flow_imshow(cotg,estacao,'\out','Vazao')
    
dict_stations = {"66120000" : ["Porto Conceição", "Rio Paraguai"]}
    #"66010000": ["Barra do Bugres","Rio Paraguai"],
    #"66072000": ["Porto Esperidião","Rio Jauru"],
    #"66090000": ["Descalvados","Rio Paraguai"],
    #"66120000": ["Porto Conceição","Rio Paraguai"],
    #"66065000": ["Estrada MT-125","Rio Cabaçal"],
    #"66055000": ["São José do Sepotuba","Rio Sepotuba"],
    #"66750000": ["Porto do Alegre","Rio Cuiabá"],
    #"66260001": ["Cuiabá","Rio Cuiabá"],
    #"66140000": ["Marzagão","Rio Paraguai"],
    #"66825000": ["Ladário","Rio Paraguai"]
    #}

print_stagedischarge(estacoes_selecionadas_str,vazao_dict,cota_dict,descarga_dict)

#print_linearseries(descarga_dict, dict_stations_rivers, 'M')
        
#corrilations = print_correlogram(vazao_dict, dict_stations)
       
#mk_values = call_mk_test(vazao_dict, dict_stations,corrilations)
    