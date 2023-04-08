import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import geopandas as gpd
import pycountry as pc # A library that provides an ISO database for all countries
import pyproj

csv_path = './DATA101_FILES/'

mismanaged_waste = pd.read_csv(csv_path + 'mismanaged_plasticwaste.csv')
ocean_plasticwaste = pd.read_csv(csv_path + 'share-of-global-plastic-waste-emitted-to-the-ocean.csv')


alpha3ISO = pd.read_excel(csv_path + 'alpha3ISO.xls') 
alpha3ISO.head()
alpha3ISO.rename(columns={'Definition':'Country'}, inplace = True)

regions = ['Africa', 'Asia', 'EU-27', 'Europe', 'Micronesia','North America', 'Oceania', 'South America']
country_ocean_plasticwaste = ocean_plasticwaste

for region in regions:
  country_ocean_plasticwaste.drop(country_ocean_plasticwaste[country_ocean_plasticwaste['Entity'] == region].index, axis = 0, inplace = True)

# Renaming Macau to it's english spelling, "Macao" to follow the format of the introduced dataset
country_ocean_plasticwaste.loc[country_ocean_plasticwaste[country_ocean_plasticwaste['Entity'] == 'Macau'].index, 'Entity'] = 'Macao'
country_ocean_plasticwaste.rename(columns = {'Entity':'Country'}, inplace = True)

null_countries = country_ocean_plasticwaste.loc[country_ocean_plasticwaste['Code'].isna() == True]
null_countries = null_countries.drop(columns = ['Code'], axis = 1)
null_countries = null_countries.merge(alpha3ISO, how = 'left', on = 'Country')
null_countries.rename(columns = {'Code Value' : 'Code'}, inplace = True)


country_ocean_plasticwaste.dropna(axis = 0, subset = ['Code'], inplace = True)
country_ocean_plasticwaste = pd.concat([country_ocean_plasticwaste,null_countries], ignore_index = True).sort_values(by=['Country'])

x = pd.merge(mismanaged_waste, alpha3ISO, on = 'Country', how = 'left')

x.rename(columns = {'Code Value': 'Code_Value'}, inplace = True)

# Seperate the null values from the x dataframe and drops the null values in
# dataframe x

y = x.loc[x['Code_Value'].isna() == True]
x.dropna(inplace = True)

def fill_code(country):
  try:
    # A try catch block that finds that returns the
    #  alpha 3 ISO code of a country.
      return pc.countries.search_fuzzy(country)[0].alpha_3
  except:
    print(country + ' not found')


y['Country'].replace('Macau', 'Macao', regex = True, inplace = True)

# Apply the fill_code function to the country attribute to fill in
# the find the missing alpha 3 ISO code and return the results to the
# 'Code_Value' attribute
y['Code_Value'] = y['Country'].apply(fill_code)

# Drop the remaining null values because they are not considered as counries.
y.dropna(inplace=True)


mismanaged_waste = pd.concat([x,y], ignore_index = True)
mismanaged_waste.reindex(columns = ['Country',
                     'Code_Value',
                     'Total_MismanagedPlasticWaste_2010 (millionT)',
                     'Total_MismanagedPlasticWaste_2019 (millionT)',
                     'Mismanaged_PlasticWaste_PerCapita_2010 (kg per year) ',
                     'Mismanaged_PlasticWaste_PerCapita_2019 (kg per year) '])


worldmap = gpd.read_file(csv_path + 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
worldmap.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

#Removes columns that have null values and unnecessary for the study
worldmap.dropna(inplace= True, axis = 1)

# Drops the 'Country' attribbute, to use the worldmap dataframe's country name
mismanaged_waste.drop(columns =['Country'], inplace = True) 

#Only get the necessary attributes in the worldmap
z = worldmap[['NAME','ISO_A3','CONTINENT','geometry']].rename(
    columns = {'ISO_A3': 'Code_Value', 'NAME':'Country'})

# Performs a right merge to keep the worldmap's shape
q = z.merge(mismanaged_waste, on ='Code_Value', how = 'left', sort = True)

# Rearranging the columns and sorting the dataframe by countryname
q = q.reindex(columns = ['Country', 'CONTINENT', 'Code_Value',
                         'Total_MismanagedPlasticWaste_2010 (millionT)',
                         'Total_MismanagedPlasticWaste_2019 (millionT)',
       'Mismanaged_PlasticWaste_PerCapita_2010 (kg per year) ',
       'Mismanaged_PlasticWaste_PerCapita_2019 (kg per year) ',
        'geometry'])
q.sort_values(by = 'Country', inplace = True)
mismanaged_waste_gdf = q

mismanaged_waste_gdf.rename(columns = {'Total_MismanagedPlasticWaste_2010 (millionT)':'Total Mismanaged Plastic Waste 2010 (millions)',
                                       'Total_MismanagedPlasticWaste_2019 (millionT)': 'Total Mismanaged Plastic Waste 2019 (millions)',
                                       'Mismanaged_PlasticWaste_PerCapita_2010 (kg per year) ':'Mismanaged PlasticWaste PerCapita 2010 (kg per year)',
                                       'Mismanaged_PlasticWaste_PerCapita_2019 (kg per year) ':'Mismanaged PlasticWaste PerCapita 2019 (kg per year)'}, inplace = True)

ocean_plasticwaste.drop(columns = ['Country'], axis = 1, inplace = True)
z.rename(columns = {'Code_Value':'Code'}, inplace = True)
q = z.merge(ocean_plasticwaste, on='Code', how='left')


#After a merge operation, the year attribute will get converted to a floating
# point decimal number, to preserve the int data type perform an
# .astype('Int64')  to the year attribute

q['Year'].fillna(2019, inplace = True) 
q['Year'] = q['Year'].astype('Int64')

ocean_plasticwaste_gdf = q
ocean_plasticwaste_gdf.rename(columns = {'Share of global plastics emitted to ocean': 'Kg Per Person'}, inplace = True)


#Choropleth map for ocean_plastic waste, 2019.
def ocean_plasticwaste_map():
  color_scale = ['#FEF0D9', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548','#D7301F','#990000']
  fig = px.choropleth_mapbox(ocean_plasticwaste_gdf,
                            geojson = ocean_plasticwaste_gdf.geometry,
                            locations=ocean_plasticwaste_gdf.index,
                              mapbox_style="carto-positron", 
                              zoom = 1,
                            hover_name = ocean_plasticwaste_gdf['Country'],
                              color = 'Kg Per Person',
                              color_continuous_scale=color_scale,height =425)

  fig.update_layout(
      margin={"r":0,"t":30,"l":10,"b":10},
      coloraxis_colorbar={
          'title':'Kg Per Person'})
  return fig


def oceanPlasticWasteVBar():
  filtered_gdf = ocean_plasticwaste_gdf
  filtered_gdf.fillna(0, inplace = True)
  filtered_gdf.sort_values(by = 'Kg Per Person', inplace = True)
  fig = px.bar(x = filtered_gdf['Kg Per Person'], y = filtered_gdf['Country'],
              hover_name = filtered_gdf['Country'],title = 'Ocean Plastic Waste Per Country in 2019',
              labels = {'x':'Kg Per Person', 'y': 'Country'},
              height = 10000)
  fig.update_traces(marker_color = '#990000')
  return fig

def categorizeOceanPlasticWaste(continent):
  if continent == 'World':
    fig = ocean_plasticwaste_map()
  else:
    color_scale = ['#FEF0D9', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548','#D7301F','#990000']

    filtered_gdf = ocean_plasticwaste_gdf.loc[ocean_plasticwaste_gdf['CONTINENT'] == continent]
    fig = px.choropleth_mapbox(filtered_gdf,
                            geojson = filtered_gdf.geometry,
                            locations=filtered_gdf.index,
                              mapbox_style="carto-positron", 
                              zoom = 1,
                            hover_name = filtered_gdf['Country'],
                              color = 'Kg Per Person',
                              color_continuous_scale=color_scale,
                               height = 425)
    fig.update_layout(
        margin={"r":0,"t":30,"l":10,"b":10},
        coloraxis_colorbar={
            'title':'Kg Per Person'})
  return fig

def display_mismanagedWaste_map(year):
  color_scale = ['#F1EEF6','#D0D1E6', '#A6BDDB','#74A9CF','#3690C0','#0570B0','#034E7B']
  if year == 2010:
    fig = px.choropleth_mapbox(mismanaged_waste_gdf,
                              geojson = mismanaged_waste_gdf.geometry,
                              locations = mismanaged_waste_gdf.index,
                              color = 'Total Mismanaged Plastic Waste 2010 (millions)',
                              hover_name = mismanaged_waste_gdf['Country'],
                              color_continuous_scale = color_scale,
                              zoom = 1, mapbox_style = 'carto-positron',height = 425)
    fig.update_layout(
    margin={"r":0,"t":30,"l":10,"b":10},
    coloraxis_colorbar={
        'title':'Total Mismanaged Plastic Waste (millions)'})
    
  else:
    fig = px.choropleth_mapbox(mismanaged_waste_gdf,
                              geojson = mismanaged_waste_gdf.geometry,
                              locations = mismanaged_waste_gdf.index,
                              color = 'Total Mismanaged Plastic Waste 2019 (millions)',
                              hover_name = mismanaged_waste_gdf['Country'],
                              color_continuous_scale = color_scale,
                              zoom = 1, mapbox_style = 'carto-positron',height = 425)
    fig.update_layout(
    margin={"r":0,"t":30,"l":10,"b":10},
    coloraxis_colorbar={
        'title':'Total Mismanaged Plastic Waste (millions)'})
  return fig

def sortMap(year,continent):

  if continent == 'World':
    fig = display_mismanagedWaste_map(year)

  else:
    color_scale = ['#F1EEF6','#D0D1E6', '#A6BDDB','#74A9CF','#3690C0','#0570B0','#034E7B']

    filtered_gdf = mismanaged_waste_gdf.loc[mismanaged_waste_gdf['CONTINENT'] == continent]
    if year == 2010:
      fig = px.choropleth_mapbox(filtered_gdf,
                                geojson = filtered_gdf.geometry,
                                locations = filtered_gdf.index,
                                color = 'Total Mismanaged Plastic Waste 2010 (millions)',
                                hover_name = filtered_gdf['Country'],
                                color_continuous_scale = color_scale,
                                zoom = 1, mapbox_style = 'carto-positron',height = 425)
      fig.update_layout(
      margin={"r":0,"t":30,"l":10,"b":10},
      coloraxis_colorbar={
          'title':'Total Mismanaged Plastic Waste (millions)'})
      
    else:
      fig = px.choropleth_mapbox(filtered_gdf,
                                geojson = filtered_gdf.geometry,
                                locations = filtered_gdf.index,
                                color = 'Total Mismanaged Plastic Waste 2019 (millions)',
                                hover_name = filtered_gdf['Country'],
                                color_continuous_scale = color_scale,
                                zoom = 1, mapbox_style = 'carto-positron',height = 425)
      fig.update_layout(
      margin={"r":0,"t":30,"l":10,"b":10},
      coloraxis_colorbar={
          'title':'Total Mismanaged Plastic Waste (millions)'})
  return fig


def mismanagedWasteVBar(year):
  bar_color = '#034E7B'
  filtered_gdf = mismanaged_waste_gdf.fillna(0)
  if year == 2010:
    filtered_gdf.sort_values(by ='Total Mismanaged Plastic Waste 2010 (millions)', inplace = True)
    fig = px.bar(x = filtered_gdf['Total Mismanaged Plastic Waste 2010 (millions)'], y =filtered_gdf['Country'],
                 title = 'Total Mismanaged Plastic Waste Per Country in 2010(millions)', hover_name = filtered_gdf['Country'],
                 labels = {'x':'Total Mismanaged Plastic Waste (millions)', 'y':'Country'},
                 height = 10000)
  else:
    filtered_gdf.sort_values(by ='Total Mismanaged Plastic Waste 2019 (millions)', inplace = True)
    fig = px.bar(x = filtered_gdf['Total Mismanaged Plastic Waste 2019 (millions)'], y =filtered_gdf['Country'],
                 title = 'Total Mismanaged Plastic Waste Per Country in 2019(millions)', hover_name = filtered_gdf['Country'],
                 labels = {'x':'Total Mismanaged Plastic Waste (millions)', 'y':'Country'},
                 height = 10000)
    
  fig.update_traces(marker_color = bar_color)
  return fig

def displayDonutGraph(category):

  categorized_gdf = mismanaged_waste_gdf.dropna()
  categorized_gdf.drop(columns = ['geometry'], inplace = True)
  categorized_gdf = categorized_gdf.groupby('CONTINENT').sum()
  categorized_gdf.reset_index(inplace = True)

  fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
  if category == 'Total':
    fig.add_trace(go.Pie(labels=categorized_gdf['CONTINENT'], values=categorized_gdf['Total Mismanaged Plastic Waste 2010 (millions)'],
                         title = '2010',name="Total Mismanaged Plastic Waste 2010 (millions)"),
                  1, 1)
    fig.add_trace(go.Pie(labels=categorized_gdf['CONTINENT'], values=categorized_gdf['Total Mismanaged Plastic Waste 2019 (millions)'],
                         title = '2019', name="Total Mismanaged Plastic Waste 2010 (millions)"),
                  1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4)


  elif category == 'Per Capita':
    fig.add_trace(go.Pie(labels=categorized_gdf['CONTINENT'], values=categorized_gdf['Mismanaged PlasticWaste PerCapita 2010 (kg per year)'], 
                         title = '2010', name='Mismanaged PlasticWaste PerCapita 2010 (kg per year)'),
                  1, 1)
    fig.add_trace(go.Pie(labels=categorized_gdf['CONTINENT'], values=categorized_gdf['Mismanaged PlasticWaste PerCapita 2019 (kg per year)'], 
                         title = '2019', name='Mismanaged PlasticWaste PerCapita 2019 (kg per year)'),
                  1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4)
  return fig

def merged_gdf_bar():

  mismanagedWaste_filtered_df = pd.DataFrame(mismanaged_waste_gdf[['Country','Code_Value','Total Mismanaged Plastic Waste 2019 (millions)']])
  mismanagedWaste_filtered_df.rename(columns = {'Code_Value':'Code'}, inplace = True)
  mismanagedWaste_filtered_df.sort_values(by = 'Code', inplace = True)

  oceanplasticwaste_filtered_df = pd.DataFrame(ocean_plasticwaste_gdf[['Code', 'Kg Per Person']])
  oceanplasticwaste_filtered_df.sort_values(by = 'Code', inplace = True)
  
  filtered_df = pd.merge(mismanagedWaste_filtered_df,oceanplasticwaste_filtered_df, on = 'Code', how = 'inner')
  filtered_df.drop_duplicates(inplace = True)
  filtered_df.fillna(0, inplace = True)

  filtered_df.sort_values(by =['Kg Per Person','Total Mismanaged Plastic Waste 2019 (millions)'], ascending = False, inplace = True) 
  filtered_df['Total Mismanaged Plastic Waste 2019 (millions)'] = filtered_df['Total Mismanaged Plastic Waste 2019 (millions)'].apply(lambda x: x / 1000000) # To scale down the graph
  fig = go.Figure()
  fig.add_trace(
      go.Bar(name = 'Kg Per Person', x = filtered_df['Country'], y = filtered_df['Kg Per Person'], marker_color = '#74A9CF')
  )
  fig.add_trace(
      go.Bar(name = 'Total Mismanaged Plastic Waste', x = filtered_df['Country'], y = filtered_df['Total Mismanaged Plastic Waste 2019 (millions)'],
             marker_color = '#3690C0')
  )
  return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    
    #Main Graph
    html.H1('Global Plastic Pollution', style ={'font-weight':'bold',
                                                'text-align':'center',
                                                'margin': '25px 0px',
                                                'padding-bottom':'20px',
                                                'border-bottom':'1px solid black'}),
    html.Div([
      html.H5('Metric:'),
      dcc.Dropdown(
          ['Mismanaged Plastic Waste', 'Share of Global Plastic Waste'],
          'Mismanaged Plastic Waste',
      style = {'width' : '500px'}, id = 'metric_dropdown', clearable = False),
    ], style = {'width': '700px', 'margin-left': '50px', 'margin-bottom':'30px'}),
      
      html.Div([
              html.H2('Mismanaged Plastic Waste Around the World', id = 'main_graph_title', style = {'width':'80%', 'word-wrap':'break-word'}),
              html.P("The data represents the sum of material which is either littered or inadequately disposed.", id = 'main_graph_desc', style = {'width':'80%', 'word-wrap':'break-word'},),  
      ], style = {'display':'flex', 'flex-direction':'column', 'align-items':'center'}),

      dbc.Container(
          html.Div(
          [
              dbc.Tabs(
                  [
                      dbc.Tab(label="Map", tab_id="map"),
                      dbc.Tab(label="Chart", tab_id="chart"),
                  ],
                  id="tabs",
                  active_tab="map",
              ),
              html.Div([
                  html.Div([dcc.Dropdown(['World','Africa','Asia','Oceania','North America', 'Seven seas (open ocean)','Europe', 'South America'],'World', 
                                         style = {'width': '250px', 'text-align': 'center', 'margin-right': '20px'}, id = 'continent_dd',clearable = False)],
                           style = {'margin-top': '30px', 'display':'flex', 'justify-content': 'flex-end', 'width': '100%'}, id = 'continent_container'),
                        
                  dcc.Graph(id = 'graph_display', figure =display_mismanagedWaste_map(2010))

              ], style = {'border': '1px solid black','height':'500px','overflow-y':'auto'}),

              html.Div([
                    html.Span('Year:', style = {'font-size': 'large'}),
                    dcc.Dropdown(['2010','2019'],'2010', style = {'width': '100px', 'text-align': 'center', 'margin-right': '20px'}, id = 'year_dd',clearable = False)
                    ],style = {'margin-top': '7px', 'display':'flex', 'justify-content': 'flex-end', 'width': '100%', 'margin-bottom': '100px'}, id = 'year_container')
            
          ]),
          className = "w-75 p-3", style = {'margin-bottom': '90px', 'border-bottom': '1px solid black'}
      ),

#Sub Graphs

      dbc.Container(
          html.Div([
              #Sub Graph 1
              html.Div([
                  html.H4('Mismanaged Plastic Waste Per Continent, 2010 vs 2019', style = {'text-align':'center'}),
                  dcc.Graph(figure = displayDonutGraph('Total'), id = 'donut_chart', style = {'width':'700px'},),
                  html.Div([
                      dcc.RadioItems(['Total', 'Per Capita'], 'Total', inline=True, inputStyle={"margin-left": "20px"}, id = 'mismanaged_plastic_waste_continent_input')                    
                  ], style = {'text-align':'center'})
                ]),
              # Sub Graph 2
              html.Div([
                  html.H4(('Total Mismanaged Waste and Plastic Pollution Per Country, 2019'),style = {'text-align':'center'}),
                  html.P(('The data represents the total mismanaged waste (millions), and the annual estimate of plastic polluton emitted in the ocean (kg per person) in 2019'),style = {'text-align':'center', 'word-wrap':'break-word'}),
                  
                  html.Div([
                    dcc.Graph(figure =merged_gdf_bar(), style = {'width' : '9000px', 'height':'500px'}) 
                  ], style = {'overflow-x':'auto'})

              ], style = {'border-left':'1px solid black','overflow-x':'auto'})

          ], className='d-flex justify-content-around'),

      className = "w-100 p-3"),
])



@app.callback(
    Output("graph_display", 'figure'),
    [Input("tabs", "active_tab"),
     Input("metric_dropdown", 'value'),
     Input('continent_dd', 'value'),
     Input('year_dd', 'value')])
def switch_tab(at,metric,continent,year):
    if at == "map":
      if metric == 'Share of Global Plastic Waste':
        fig = categorizeOceanPlasticWaste(continent)
        return fig
      elif metric == 'Mismanaged Plastic Waste':
        fig = sortMap(int(year),continent)
        return fig
    elif at == "chart":
      if metric == 'Mismanaged Plastic Waste':
        if int(year) == 2010:
          fig= mismanagedWasteVBar(2010)
        else:
          fig = mismanagedWasteVBar(2019)
        return fig
      else:
        fig = oceanPlasticWasteVBar()
        return fig

@app.callback(
    Output('year_container', 'style'),
    Input('metric_dropdown', 'value')
)
def yearVisibility(metric):
  if metric == 'Share of Global Plastic Waste':
    return {'display':'none'}
  return {'margin-top': '7px', 'display':'flex', 'justify-content': 'flex-end', 'width': '100%'}

@app.callback(
    Output('continent_container', 'style'),
    Input('tabs', 'active_tab')
)
def continentVisibility(at):
  if at == 'map':
    return {'margin-top': '30px', 'display':'flex', 'justify-content': 'flex-end', 'width': '100%'}
  else:
    return {'display':'none'}

@app.callback(
    [Output('main_graph_title', 'children'), 
     Output('main_graph_desc', 'children')],
    Input('metric_dropdown', 'value')
)
def modifyMainGraphTitle(metric):
  title = ['Share of Global Plastic Waste Emitted to the Ocean, 2019',
           'Mismanaged Plastic Waste Around the World']
  desc = ['The data is an annual estimate of plastic emission around the world.',
          'The data represents the sum of material which is either littered or inadequately disposed.']
  if metric == 'Mismanaged Plastic Waste':
    return (title[1], desc[1])
  else:
    return (title[0], desc[0])

@app.callback(
    Output('donut_chart', 'figure'),
    Input('mismanaged_plastic_waste_continent_input', 'value')
)
def modifyDonutChart(metric):
  if metric == 'Total':
    return displayDonutGraph('Total')
  else:
    return displayDonutGraph('Per Capita')
# Run the server
if __name__ == '__main__':
    app.run_server()
    


