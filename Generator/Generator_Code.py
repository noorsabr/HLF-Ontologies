import nbformat  # for creating Jupyter notebooks
from owlready2 import *  # for working with OWL ontologies

# load the ontology file
onto = get_ontology("./FabricDeploymentOntology").load()

def get_class(ontology, class_name: str):
    classes = list(ontology.classes())
    [single_class] = [c for c in classes if c.name == class_name]
    return single_class

# Define the base classes from the ontology for later use
comp = get_class(onto, "CompositeActivity")
atom = get_class(onto, "AtomicActivity")
seq = get_class(onto, "SequentialActivity")
fork = get_class(onto, "ForkedActivity")
act = get_class(onto, "Activity")
alt = get_class(onto, "AlternatingActivity")

base_classes = [comp, atom, seq, fork, act, alt]

# Function to get all individuals that has parent and sub-activity (instances)
def get_individuals(cls):
    individuals = []
    for i in cls.instances():
        ind_dict = {"name": i.label.first()}
        if i.hasParentActivity:
            ind_dict["hasParentActivity"] = i.hasParentActivity[0].label.first()
        if i.hasSubactivity:
            subactivities = []
            for sub in i.hasSubactivity:
                subactivities.append(sub.label.first())
            ind_dict["hasSubactivity"] = subactivities

        # Get all data property values for the individual
        data_dict = {}
        for prop in i.get_properties():
            if isinstance(prop, DataPropertyClass):
                value = getattr(i, prop.python_name, None)
                if value is not None:
                    data_dict[prop.label[0]] = value
        ind_dict["data_properties"] = data_dict
        individuals.append(ind_dict)
    return individuals

# Function to get all descendants of a class
def get_true_des(cls):
    all_des = cls.descendants()
    all_des.remove(cls)
    return all_des

# Function to get all leaves (instances without subactivities) of activity classes
def get_activity_leaves(ontology):
    act_classes = act.descendants().difference(base_classes)
    leaves = {}

    for cls in act_classes:
        if len(get_true_des(cls)) == 0:
            individuals = get_individuals(cls)
            leaves[cls.name] = individuals
    return leaves

# Function to get the hierarchy of activities using DFS traversal
def get_activity_hierarchy_DFS(ontology):
    root_activities = [a for a in act.instances() if not a.hasParentActivity]
    hierarchy = {}
    for root in root_activities:
        stack = [(root, None, 1)]
        while stack:
            node, parent, level = stack.pop()
            if node in hierarchy:
                hierarchy[node]["level"] = min(hierarchy[node]["level"], level)
            else:
                hierarchy[node] = {"level": level, "children": []}
            if parent:
                hierarchy[parent]["children"].append(node)
            children = node.hasSubactivity
            if children:
                sort_activities_by_metBy(children)
                for child in reversed(children):
                    stack.append((child, node, level + 1))
    for node in hierarchy:
        hierarchy[node]["children"].sort(key=lambda x: node.hasSubactivity.index(x))
    return hierarchy

# Function to sort a list of activities by their 'metBy' attribute
def sort_activities_by_metBy(activities):
    activities.sort(key=lambda x: x.metBy)

# Get the activity hierarchy from the ontology using DFS traversal
activity_hierarchy = get_activity_hierarchy_DFS(onto)
root_activities = [a for a in activity_hierarchy if activity_hierarchy[a]["level"] == 1]
for root in root_activities:
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        children = node.hasSubactivity
        if children:
            sort_activities_by_metBy(children)
            for child in reversed(children):
                stack.append((child, node))

        # Print activity name and hierarchy
        activity_name = node.label.first()
        print("  " * (activity_hierarchy[node]["level"] - 1) + "- " + activity_name)


# Generate a Jupyter notebook for the main activity hierarchy
main_notebook = nbformat.v4.new_notebook()
main_notebook.cells = []

# Get the activity hierarchy from the ontology using DFS traversal
activity_hierarchy = get_activity_hierarchy_DFS(onto)

# Generate cell for importing libraries
libraries_code = '''\
import plotly.express as px
import hvplot.pandas
import holoviews as hv
import plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from holoviews.operation.datashader import datashade
from datetime import datetime, timezone
from plotly.subplots import make_subplots'''
main_notebook.cells.append(nbformat.v4.new_code_cell(libraries_code))

# Generate cell for extension library
extention_code = '''\
hvplot.extension('plotly', 'bokeh')'''
main_notebook.cells.append(nbformat.v4.new_code_cell(extention_code))

# Generate cell for offline rendering
offline_code = '''\
import plotly.io as pio
pio.renderers.default = "notebook"'''
main_notebook.cells.append(nbformat.v4.new_code_cell(offline_code))

# Generate cell for loading the CSV file
csv_file_location = '''\
df = pd.read_csv('./renamed.csv', engine='python')'''
main_notebook.cells.append(nbformat.v4.new_code_cell(csv_file_location))

# Generate cell for sorting the DF
df_sort_cell = '''\
sorted_df = df.sort_values(by='act_transaction_processing_beginning')'''
main_notebook.cells.append(nbformat.v4.new_code_cell(df_sort_cell))

#Generate cell for the filtering
filtering_code = '''\
start_time = datetime.strptime("2021-04-12 20:22:41.0000", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
end_time = datetime.strptime("2021-04-12 20:23:50.0000", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)

sorted_df['beginning_datetime_filter'] = pd.to_datetime(sorted_df['act_transaction_processing_beginning'], format='%Y-%m-%d %H:%M:%S.%f')

sorted_df = sorted_df[(sorted_df['beginning_datetime_filter'] >= start_time) & (sorted_df['beginning_datetime_filter'] <= end_time)].copy()'''
main_notebook.cells.append(nbformat.v4.new_code_cell(filtering_code))

# Generate cell for the top 500 tx the DF
#df_top500_cell = '''\
#top500 = sorted_df[0:500] '''
#main_notebook.cells.append(nbformat.v4.new_code_cell(df_top500_cell))

# Generate cell for histogram
histogram_code = '''\
def plot_histogram(sorted_df, column_name, title='', xlabel='', ylabel=''):
    hist_plot = sorted_df[column_name].hvplot.hist(title=title, xlabel=xlabel)
    return hist_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(histogram_code))

# Generate cell for histogram with log
hist_log_code = '''\
def plot_histogram_log(sorted_df, column_name, title='', xlabel='', ylabel=''):
    hist_plot = sorted_df[column_name].dropna()[sorted_df[column_name]> 0].hvplot.hist(logy=True, title=title, xlabel=xlabel, ylabel=ylabel)
    return hist_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(hist_log_code))

# Generate cell for Box plot
boxplot_code = '''\
def plot_boxplot(sorted_df, column_name, title='', label=''):
    box_plot = sorted_df[column_name].dropna().hvplot.box(title=title, label=label)
    return box_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(boxplot_code))

# Generate cell for Box plot with logarithmic scale
boxplot_log_code = '''\
def plot_boxplot_log(sorted_df, column_name, title='', label=''):
    sorted_df[column_name] = sorted_df[column_name].apply(lambda x: x if x > 0 else np.nan)
    box_plot = sorted_df[column_name].dropna().hvplot.box(logy=True, title=title, label=label, ylabel=label)
    return box_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(boxplot_log_code))

# Generate cell for scatter plot
scatter_code = '''\
def plot_scatter(sorted_df, x_column, y_column, title='', xlabel='', ylabel=''):
    scatter_plot = sorted_df.hvplot(x=x_column, y=y_column, kind='line', title=title)
    scatter_plot = scatter_plot.opts(title=title, xlabel=xlabel, ylabel=ylabel)
    return scatter_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(scatter_code))

# Generate cell for scatter plot with logarithmic scale
scatter_log_code = '''\
def plot_scatter_log(sorted_df, x_column, y_column, title='', xlabel='', ylabel=''):
    scatter_plot = sorted_df.hvplot(x=x_column, y=y_column, kind='line', title=title, logy=True)
    scatter_plot = scatter_plot.opts(title=title, xlabel=xlabel, ylabel=ylabel)
    return scatter_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(scatter_log_code))

# Generate cell for scatter plot with datashading
scatter_shade_code = '''\
def plot_scatter_shade(sorted_df, x_column, y_column, title='', xlabel='', ylabel=''):
    sorted_df[x_column] = pd.factorize(sorted_df[x_column])[0]
    scatter_plot = sorted_df.hvplot(x=x_column, y=y_column, kind='line', title=title, datashade=True)
    scatter_plot = scatter_plot.opts(title=title, xlabel=xlabel, ylabel=ylabel)
    return scatter_plot'''
main_notebook.cells.append(nbformat.v4.new_code_cell(scatter_shade_code))


# Generate cell for stack of scatter plots (Function to create a single plot with shared x-axis for sequential activity and its children)
plot_stack_scatter_code = '''\
def plot_combined_scatter(sorted_df, x_column, y_columns, y_labels, titles, xlabel='', ylabel='', width=800):
    scatter_plots = []
    x = sorted_df[x_column]
    for idx, (y_column, y_label) in enumerate(zip(y_columns, y_labels)):
        y = sorted_df[y_column]
        scatter_plot = hv.Curve((x, y), label=y_label).opts(title=titles[idx], xlabel=xlabel, ylabel=ylabel, width=width)
        scatter_plots.append(scatter_plot)
    combined_plot = hv.Layout(scatter_plots).cols(1)
    return combined_plot.opts(shared_axes=True)'''
main_notebook.cells.append(nbformat.v4.new_code_cell(plot_stack_scatter_code))

""" parallel_plot_code = '''\
def plot_parallel_coordinates(df, duration_columns, child_labels, title=''):
    data = pd.DataFrame()
    for column, (label, _) in zip(duration_columns, child_labels):
        data[label] = df[column]
    return data.hvplot.parallel_coordinates(class_column='class', width=800, title=title)
'''
main_notebook.cells.append(nbformat.v4.new_code_cell(parallel_plot_code)) """

# Generate a Jupyter notebook for each activity hierarchy
for activity, hierarchy_info in activity_hierarchy.items():
    activity_name = activity.label.first()
    duration_individuals = activity.hasDuration
    start_individuals = activity.hasStart
    child_durations = activity.hasSubactivity

    duration_column_name = None
    start_column_name = None
    has_display_name_duration = None
    has_display_name_start = None

# Find the column name for duration
    for individual in duration_individuals:
        if individual.hasColumn:
            column_individual = individual.hasColumn[0]
            if column_individual.hasColumnName:
                duration_column_name = column_individual.hasColumnName[0]
                has_display_name_duration = column_individual.hasDisplayName[0]
                print(has_display_name_duration)
                break

# Find the column name for start
    for individual in start_individuals:
        if individual.hasColumn:
            column_individual = individual.hasColumn[0]
            if column_individual.hasColumnName:
                start_column_name = column_individual.hasColumnName[0]
                has_display_name_start = column_individual.hasDisplayName[0]
                print(has_display_name_start)
                break

    # Create a new notebook for the activity hierarchy
    activity_notebook = nbformat.v4.new_notebook()
    activity_notebook.cells = []


        
    # Cell for histogram
    if duration_column_name is not None:    
        histogram_cell_code = f"plot_histogram(sorted_df, '{duration_column_name}', title='{has_display_name_duration}', xlabel='Duration (ms)')"
        
    else:
        histogram_cell_code = "# Unable to create a Histogram plot because of missing activity Duration data."
        

    # Cell for histogram with logarithmic scale
    if duration_column_name is not None:    
        histogram_log_cell_code = f"plot_histogram_log(sorted_df, '{duration_column_name}', title = 'Duration of {activity.label.first()} activity (log)', xlabel='Duration (ms)')"
        
    else:
        histogram_log_cell_code = "# Unable to create a Histogram plot because of missing activity Duration data."
        

     # Cell for Box plot
    if duration_column_name is not None:    
        boxplot_cell_code = f"plot_boxplot(sorted_df, '{duration_column_name}', title='{has_display_name_duration}', label='Duration (ms)')"
        
    else:
        boxplot_cell_code = "# Unable to create a Box plot because of missing activity Duration data."
        

    # Cell for Box plot with logarithmic scale
    if duration_column_name is not None:
        boxplot_log_cell_code = f"plot_boxplot_log(sorted_df, '{duration_column_name}', title = 'Duration of {activity.label.first()} (log)', label='Duration (ms)')"
        
    else:
        boxplot_log_cell_code = "# Unable to create a Box plot because of missing activity Duration data."
           

    # Cell for scatter plot
    if duration_column_name is not None and start_column_name is not None :
        scatter_cell_code = f"plot_scatter(sorted_df, '{start_column_name}', '{duration_column_name}', title='Duration timeseries of {activity_name}', xlabel= '{has_display_name_start}', ylabel='{has_display_name_duration}')"
        
    else: 
        scatter_cell_code = "# Unable to create a Scatter plot because of missing activity Beginning and Duration data."
       

    # Cell for scatter plot with logarithmic scale
    if duration_column_name is not None and start_column_name is not None :
        scatter_log_cell_code = f"plot_scatter_log(sorted_df, '{start_column_name}', '{duration_column_name}', title='Duration timeseries of {activity_name} (log)', xlabel= '{has_display_name_start}', ylabel='{has_display_name_duration}')"
        
    else: 
        scatter_log_cell_code = "# Unable to create a Scatter plot because of missing activity Beginning and Duration data."
        

    # Cell for scatter plot with datashading
    if duration_column_name is not None and start_column_name is not None:
        scatter_shade_cell_code = f"plot_scatter_shade(sorted_df, '{start_column_name}', '{duration_column_name}', title='Duration timeseries of {activity_name} (datashading)', xlabel= '{has_display_name_start}', ylabel='{has_display_name_duration}')"
        
    else:
        scatter_shade_cell_code = "#Unable to generate Scatter Plot using Datashading because of missing activity Beginning and Duration data."
        


# Get the children column names
    child_labels = []
    child_durations = []
    child_y_columns = []
    child_y_labels = []
    child_titles = []
    for individual in activity.hasSubactivity:
            if individual.hasDuration:
                durations_individual = individual.hasDuration[0]
                if durations_individual.hasColumn:
                    column_individual = durations_individual.hasColumn[0]
                    if column_individual.hasColumnName:
                        duration = column_individual.hasColumnName[0]
                        child_durations.append(duration)
                        label_with_comment = f" Duration of {individual.label.first()} (ms)"
                        child_labels.append((individual.label.first(), label_with_comment))
                        child_y_columns.append(duration)
                        child_y_labels.append(label_with_comment)
                        child_titles.append(f"{activity_name} - {individual.label.first()}")
                    else:
                        print("No column name found for individual:", individual.label.first())
                else:
                    print("No duration column found for individual:", individual.label.first())
            else:
                print("No duration found for individual:", individual.label.first())

# Cell for Parallel coordinates
    if isinstance(activity, seq):
        labels = [f"{comment}" for label, comment in child_labels]
        para_cell_code = f'''
df2 = sorted_df.copy()
df2['class'] = 'Parent activity and sub-activities duration (ms)'
df2 = df2.sample(frac=0.01)
parent_duration = ['{duration_column_name}']
children_durations = {child_y_columns}
cols = parent_duration + children_durations
#labels = {{'{duration_column_name}': '{has_display_name_duration}', **{{col: label for col, label in zip({child_y_columns}, {child_y_labels})}}}}
hvplot.plotting.parallel_coordinates(df2, class_column='class', cols=cols, width=800)
'''
        
    else:
        print("Not a Sequential Activity! ")
        para_cell_code = "# Not a Sequential Activity!"
        

# Cell for box plots combined 
    if isinstance(activity, seq):
        labels = [f"{comment}" for label, comment in child_labels]
        combined_box_cell_code = f'''#Box plots 
parent_duration = ['{duration_column_name}']
children_durations = {child_y_columns}
children_labels = {child_y_labels}
cols = parent_duration + children_durations
df_to_plot = sorted_df[cols].melt()
fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
labels = {{'{duration_column_name}': '{has_display_name_duration}', **{{col: label for col, label in zip(children_durations, children_labels)}}}}

# Add box plots for each column to the subplots
for i, col in enumerate(df_to_plot['variable'].unique(), 1):
    x_values = [labels[col]] * len(df_to_plot.loc[df_to_plot['variable'] == col, 'value'])  # Repeat the x-axis label for each box trace
    fig.add_trace(go.Box(y=df_to_plot.loc[df_to_plot['variable'] == col, 'value'], x=x_values, name=labels[col]), row=1, col=i)

fig.update_layout(title="Comparision between sub-activites of Transaction Processing Activity", xaxis_title="Variables", yaxis_title="Duration (ms)", legend=dict(bgcolor='white'), plot_bgcolor='white')

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.show()'''
        
    else:
        print("Not a Sequential Activity! ")
        combined_box_cell_code = "# Not a Sequential Activity!"
        


 # Cell for scatter plots stack
    if isinstance(activity, seq) and duration_column_name is not None and start_column_name is not None and len(child_y_columns) > 0:
    #if duration_column_name is not None and start_column_name is not None and len(child_y_columns) > 0:
        combined_scatter_cell_code = f"plot_combined_scatter(sorted_df, '{start_column_name}', ['{duration_column_name}'] + {child_y_columns}, ['Sequential Activity'] + {child_y_labels}, ['{activity_name}'] + {child_titles}, xlabel='{has_display_name_start}', ylabel='Duration (ms)')"
        

    else:
        combined_scatter_cell_code = "# Unable to create Scatter plots because of missing activity Beginning and Duration data."
        

    
    # Markdown cell for header (activity hierarchy name)
    activity_notebook.cells.append(nbformat.v4.new_markdown_cell("#" * activity_hierarchy[activity]["level"] + " " + "<span style=\"color:red\">" + activity.label.first() + "</span>"))


   # Markdown and code for each visualization type
    visualization_1 = [
        ("Histogram", histogram_cell_code),
        ("Histogram with log", histogram_log_cell_code),
        ("Box Plot", boxplot_cell_code),
        ("Box Plot with log", boxplot_log_cell_code),
        ("Scatter Plot", scatter_cell_code),
        ("Scatter Plot with log", scatter_log_cell_code),
        ("Scatter Plot with data shading", scatter_shade_cell_code)
        ]


    visualization_2 = [
        ("Parallel Coordinates", para_cell_code),
        ("Combined Scatter Plots", combined_scatter_cell_code),
        ("Combined Box Plots", combined_box_cell_code)
    ]

   # Grouping Markdown cells for the first category
    markdown_description = "#### Marginal Properties"
    activity_notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_description))

    # Append visualization codes to activity's Markdown cell
    for vis_type, vis_code in visualization_1:
        markdown_description = f"##### {vis_type}"
        activity_notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_description))
        activity_notebook.cells.append(nbformat.v4.new_code_cell(vis_code))

    # Grouping Markdown cells for the second category
    markdown_description2 = "#### Drill-Down Plots"
    activity_notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_description2))

    # Append visualization codes to activity's Markdown cell
    for vistype, viscode in visualization_2:
        markdown_description2 = f"##### {vistype}"
        activity_notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_description2))
        activity_notebook.cells.append(nbformat.v4.new_code_cell(viscode))


    # Add the cells from the activity hierarchy notebook to the main notebook
    main_notebook.cells.extend(activity_notebook.cells)

# Save the main notebook
nbformat.write(main_notebook, "Generated_Notebook.ipynb")   