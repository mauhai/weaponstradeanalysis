''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import pandas as pd
import numpy as np
import os
import networkx as nx
import operator
import matplotlib.pyplot as plt
import community
import math
import copy


from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

# Set up data

# coding: utf-8



# In[3]:

def loadTable(directory, country_code):
    path = directory+'TIV-Import-'+country_code+'-1950-2015.csv'
    df = pd.read_csv(path)
    
    #extract current country
    to_country = df.columns.values.tolist()[0].split('TIV of arms exports to ')[1].split(', 1950-2015')[0]
    
    #downsize
    df = df.ix[9:]
    
    #get rid of column
    df = df.drop(df.columns[[0]],axis =1)
    df.columns = df.iloc[0]
    df.columns.values[0] = 'country'
    df = df.set_index((df['country']))
    df = df.drop(df.columns[0], axis=1)
    
    #take the data less the header row
    df = df[1:] 
    df.index.name = None #credit to ulynn
    df.columns.name = None
    
    # Format of the keys should be YEAR - COUNTRYFROM - COUNTRYTO --> Faster query over the years

    df.columns = df.columns.astype('str').str.replace('\.0','')
        
    df.fillna(0, inplace=True)
    try:
        df.drop(df.loc['Unknown country'].name,inplace=True)
    except:
        ""
        
    #last cleansing
    df.drop(df.index[[-1,-2]],inplace=True)
    df.drop(df.columns[-1], axis=1,inplace=True)

    return df, to_country


# In[4]:

#FORMAT
#[YEARS][FROM][TO] = MONEY
def convertTableToDict(df, onecountrydict, countryTo):
    years = list(df.columns.values)
    countries = list(df.index)
    
    for year in years:
        onecountrydict.setdefault(year, dict())
        onecountrydict[year].setdefault(countryTo, dict())
        
        for country in countries:
            value = df.get_value(country, year)
            onecountrydict[year][countryTo].setdefault(country, value)
            
    return onecountrydict


# In[5]:

#delete empty cells from dict

def clean_empty(d):
    if not isinstance(d, (dict, list)): #dictionairy or list
        return d
    if isinstance(d, list): 
        return [v for v in (clean_empty(v) for v in d) if v] #list comprehension
    return {k: v for k, v in ((k, clean_empty(v)) for k, v in d.items()) if v}


# In[6]:

countryImportDict = dict()
PATH = '/Users/hai/Devproj/weaponstradeanalysis/data/'

countryCodeMap = dict()

for f in os.listdir(PATH):
    if not f.startswith('.'):
        if "country_codes.csv" not in f:
            countryCode = f.replace('TIV-Import-',"").replace('-1950-2015.csv', "")
            df,to_country = loadTable(PATH, countryCode)
            countryImportDict = convertTableToDict(df, countryImportDict, to_country)
            countryCodeMap.setdefault(to_country, countryCode)

countryImportDict = clean_empty(countryImportDict)

MultiDiDict = dict()
for year in countryImportDict:
    MultiDiDict.setdefault(year, dict())
    for countryImport in countryImportDict[year]:
        for countryExport in countryImportDict[year][countryImport]:
            MultiDiDict[year].setdefault(countryExport, dict())
            MultiDiDict[year][countryExport].setdefault(countryImport, 0)
            MultiDiDict[year][countryExport][countryImport] += countryImportDict[year][countryImport][countryExport]


# In[7]:

df = pd.DataFrame.from_dict(countryCodeMap, orient="index")
df.sort_index(inplace=True)
df.to_csv('countrymap.csv')


# In[8]:

militaryexpdf = pd.read_excel('/Users/hai/Devproj/weaponstradeanalysis/newdata/SIPRI extended milex database beta/Constant USD Beta.xlsx')
militaryexpdf = militaryexpdf.iloc[2:175, 2:]
militaryexpdf.columns = militaryexpdf.iloc[0]
militaryexpdf.columns.values[0] = 'Country'
militaryexpdf = militaryexpdf.set_index((militaryexpdf['Country']))
militaryexpdf = militaryexpdf.drop(militaryexpdf.columns[0], axis=1)
militaryexpdf = militaryexpdf[1:] 
militaryexpdf.index.name = None #credit to ulynn
militaryexpdf.columns.name = None
militaryexpdf.drop(['Montenegro'], inplace = True)


namecorrectionmapping = pd.read_csv('/Users/hai/Devproj/weaponstradeanalysis/countrymapping.csv', delimiter = ';', header = None)
namecorrectionmapping.columns = ['countrycode','newname', 'oldname']

namecorrectionmapping.set_index((namecorrectionmapping['oldname']), inplace = True)
namecorrectionmapping.drop(namecorrectionmapping.columns[[0,2]], axis=1, inplace = True)
namecorrectionmapping.index.name = None
namemappingdict = namecorrectionmapping.to_dict()


from decimal import Decimal


def convertmilitaryexpTableToDict(df,namemapping):
    years = list(df.columns.values)
    countries = list(df.index)
    militaryexpdict = dict()
    for year in years:
        militaryexpdict.setdefault(str(year), dict())
        for country in countries:
            value = df.get_value(country, year)
            if isinstance(value,float):
                value = round(Decimal(value * 0.5520917815626),1) #inflation correction from constant 2014 to constant 1990
            militaryexpdict[str(year)].setdefault((namemapping['newname'][country]), value)
    return militaryexpdict

militaryexpdict = convertmilitaryexpTableToDict(militaryexpdf, namemappingdict)


# In[9]:

excludedentities = pd.read_csv('excludedcountries.csv', delimiter=';', header = None)
excludedentities.columns = ['Name','Code']
excludedentities.set_index((excludedentities['Name']), inplace = True)
excludedentities.drop(excludedentities.columns[[0]], axis=1, inplace = True)
excludedentities.index.name = None
excludedentities = excludedentities.index


# In[10]:

def clean_noncountries(d):
    if not isinstance(d, (dict)): #dictionairy or list
        return d
    return {k: v for k, v in ((k, clean_noncountries(v)) for k, v in d.items()) if k not in excludedentities}


# In[11]:

def createNeighbourGraph(G, node):
    
    newGraph = nx.Graph()
    
    for edge in G.edges(data=True):
        if edge[0] == node or edge[1] == node:
            newGraph.add_edge(edge[0],edge[1],edge[2])
    
    return newGraph


# In[12]:

# most fucked up function

def addUpDict(MasterDict):

    SomethingsOverwrittenDict = copy.deepcopy(MasterDict)

    AddedUpDict = dict()

    for year in SomethingsOverwrittenDict:
        AddedUpDict.setdefault(year, dict())
        for countryExport in SomethingsOverwrittenDict[year]:
            for countryImport in SomethingsOverwrittenDict[year][countryExport]:
                AddedUpDict[year].setdefault(countryExport, dict())
                AddedUpDict[year][countryExport].setdefault(countryImport, 0)
                Richtung = SomethingsOverwrittenDict[year][countryExport][countryImport]
                #andere richtung verfügbar?
                if countryImport in SomethingsOverwrittenDict[year]:
                    if countryExport in SomethingsOverwrittenDict[year][countryImport]:
                        andereRichtung = SomethingsOverwrittenDict[year][countryImport][countryExport]
                    else: andereRichtung = 0
                else: andereRichtung = 0
                
                #Transmitting stuff into the AddedUpDict
                if (Richtung + andereRichtung) != 0:
                    AddedUpDict[year][countryExport][countryImport] = Richtung + andereRichtung

                #"Clearing the MasterDict"
                SomethingsOverwrittenDict[year][countryExport][countryImport] = 0

                if countryImport in SomethingsOverwrittenDict[year]:
                    if countryExport in SomethingsOverwrittenDict[year][countryImport]:
                        SomethingsOverwrittenDict[year][countryImport][countryExport] = 0
    
    return AddedUpDict


# In[13]:

def createYearGraph(AddedUpDict, year, militaryexpdict):
    
    G = nx.Graph()    
    for countryExport in AddedUpDict[year]:
        for countryImport in AddedUpDict[year][countryExport]:
            if AddedUpDict[year][countryExport][countryImport] != 0:
                G.add_weighted_edges_from([(countryExport,countryImport,AddedUpDict[year][countryExport][countryImport])])
    
    for node in G.nodes_iter():
        if node in militaryexpdict[year]:
            G.node[node]['military expenditure'] = militaryexpdict[year][node]
        else:
            G.node[node]['military expenditure'] = 'no data'
            
    G.graph['year']= year
    
    return G


# In[14]:

def createYearMultiDiGraph(AddedUpDict, year,militaryexpdict):
    
    G = nx.MultiDiGraph()    
    for countryExport in AddedUpDict[year]:
        for countryImport in AddedUpDict[year][countryExport]:
            if AddedUpDict[year][countryExport][countryImport] != 0:
                G.add_weighted_edges_from([(countryExport,countryImport,AddedUpDict[year][countryExport][countryImport])])
    
    for node in G.nodes_iter():
        if node in militaryexpdict[year]:
            G.node[node]['military expenditure'] = militaryexpdict[year][node]
        else:
            G.node[node]['military expenditure'] = 'no data'

    G.graph['year']= year

    
    return G


# # Setup Dicts

# In[15]:

MultiDiDict = clean_empty(clean_noncountries(MultiDiDict))
AddedUpDict = clean_empty(clean_noncountries(addUpDict(MultiDiDict)))

year = '1960'

multigraph = createYearMultiDiGraph(MultiDiDict, year ,militaryexpdict)
mastergraph = createYearGraph(AddedUpDict, year , militaryexpdict)


# In[16]:
from bokeh.palettes import Set1_9
colormap = Set1_9


# In[17]:

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models.widgets import Slider, TextInput

# In[23]:

def get_nodes_specs(_network, _layout):
    d = dict(x=[], y=[], color=[], country=[], size=[], totaltrade =[], 
             cluster=[], alpha=[], largestpartner=[], military_expenditure=[],year=[])
    
    partition = community.best_partition(_network)
    year = _network.graph['year']
    for node in _network.nodes(data = True):
        totaltrade = 0
        largest_trade_partner_trade = 0
        largest_trade_partner_name = ''  
        
        for edge in _network.edges(data=True):
                if edge[0] == node[0] or edge[1] == node[0]:
                    totaltrade += edge[2].values()[0]
                    if edge[2].values()[0] >= largest_trade_partner_trade:
                        largest_trade_partner_trade = edge[2].values()[0]
                        if edge[0] == node[0]:
                            largest_trade_partner_name = edge[1]
                        else:
                            largest_trade_partner_name = edge[0] 

        d['x'].append(_layout[node[0]][0])
        d['y'].append(_layout[node[0]][1])
        d['color'].append(colormap[partition[node[0]]])
        d['country'].append(node[0])
        
        if isinstance(node[1].values()[0],float) and not math.isnan(node[1].values()[0]):
            #d['size'].append((np.log(node[1].values()[0]))*0.8)
            d['alpha'].append(0.7)
        else:
            d['alpha'].append(0.2)
        
        if isinstance(np.log(totaltrade),(float,int)) and not math.isnan(np.log(totaltrade)):
            d['size'].append(2+np.log(totaltrade))
        else:
            d['size'].append(1)

        d['totaltrade'].append(totaltrade)
        d['cluster'].append(partition[node[0]])
        
        d['largestpartner'].append(largest_trade_partner_name)
        #if isinstance(node[1].values()[0],(float,int)) and not math.isnan(node[1].values()[0]):
        #    d['military_expenditure'].append(int(node[1].values()[0]))
        #else:
        #    d['military_expenditure'].append('NaN')
        d['military_expenditure'].append(10)
        d['year'].append(int(year))
        
    return d


# In[24]:

def get_edges_specs(_network, _layout):
    d = dict(xs=[], ys=[], alphas=[], width=[], year=[])
    year = _network.graph['year']

    weights = []
    for u, v, data in _network.edges(data=True):
        weights.append(data)
    max_weight = max(weights).values()
    calc_alpha = lambda h: 0.1 + 0.6 * (h / max_weight)
    for u, v, data in _network.edges(data=True):
        d['xs'].append([_layout[u][0], _layout[v][0]])
        d['ys'].append([_layout[u][1], _layout[v][1]])
        d['alphas'].append(calc_alpha(data['weight']))
        d['width'].append(np.log(data.values()[0]))
        d['year'].append(int(year))
    
    return d

#setup input controls

year = Slider(title="year", value=1950, start=1950, end=2015, step=1)

## preparing the data - loading the data

def AllGraphs(countrydict, militaryexpdict):
    AllGraphsDict = dict()
    for year in countrydict:
        G = createYearGraph(countrydict, year, militaryexpdict)
        layout = nx.spring_layout(G,scale=2)
        AllGraphsDict[year] = [G,layout]
    return AllGraphsDict

AllGraphs = AllGraphs(AddedUpDict, militaryexpdict)

## preparing the data - massaging the data

allnodesdf = pd.DataFrame()
alledgesdf = pd.DataFrame()

for dictyear in AddedUpDict:
    addnodes = pd.DataFrame.from_dict(data=get_nodes_specs(AllGraphs[dictyear][0],AllGraphs[dictyear][1]))
    allnodesdf = allnodesdf.append(addnodes)

    addedges = pd.DataFrame.from_dict(data=get_edges_specs(AllGraphs[dictyear][0],AllGraphs[dictyear][1]))
    alledgesdf = alledgesdf.append(addedges)

source = ColumnDataSource(data=dict(x=[], y=[], color=[], country=[], size=[], totaltrade =[], 
             cluster=[], alpha=[], largestpartner=[], military_expenditure=[],year=[]))
source2 = ColumnDataSource(data=dict(xs=[], ys=[], alphas=[], width=[], year=[]))

hover = HoverTool(tooltips=[('country','@country'),
                            ('cluster','@cluster'),
                            ('military_expenditure','@military_expenditure'),
                            ('largest tradepartner', '@largestpartner'),
                            #('imports from', '@import'),
                            #('exports from', '@export')
                           ])

p = figure(plot_width=500, plot_height=500, tools=['pan','tap',hover,'box_zoom','reset','wheel_zoom','save'])

p.circle('x','y', source=source, size='size', color='color', level='overlay',alpha='alpha')
p.multi_line('xs', 'ys', source=source2, line_width='width', alpha='alphas', color='navy')


def select_nodes():
    selected = allnodesdf[
        (allnodesdf.year == year.value) ]
    return selected

#Baustelle 1
def select_edges():
    selected = alledgesdf[
        (alledgesdf.year == year.value) ]
    return selected


def update():
    nodesdf = select_nodes()
    edgesdf = select_edges()
    p.title.text = "%d countries selected" % len(nodesdf)
    source.data = dict(
        x=nodesdf['x'],
        y=nodesdf['y'],
        color=nodesdf["color"],
        country=nodesdf["country"],
        year=nodesdf["year"],
        alpha=nodesdf["alpha"],
        cluster=nodesdf['cluster'],
        size=nodesdf['size'],
        military_expenditure=nodesdf['military_expenditure'],
        largestpartner=nodesdf['largestpartner']
        )
    
    # Baustelle 2
    
    source2.data = dict(
         xs=edgesdf['xs'], ys=edgesdf['ys'], alphas=edgesdf['alphas'], width=edgesdf['width'], year=edgesdf['year']
        )   
        

controls = [year]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = widgetbox(*controls, sizing_mode='fixed')
l = row(inputs, p, sizing_mode='fixed')

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "military"

## Something is with the NaN Values!!!





















