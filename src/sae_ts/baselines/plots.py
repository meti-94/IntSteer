# %%
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# %%
def main(data_path, is_2b=True, name=""):
    with open(data_path, 'r') as f:
        graph_data_list = json.load(f)

    # Get the unique steering goals and sort them case-insensitively
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list), key=str.lower)

    # Capitalize the steering goals for display
    capitalized_goals = [goal.capitalize() for goal in steering_goals]

    # Prepare a mapping from steering goal to its data
    data_by_goal = {goal: [] for goal in steering_goals}
    for data in graph_data_list:
        goal = data['steering_goal_name']
        data_by_goal[goal].append(data)

    # Create subplots: 3 columns, as many rows as needed
    num_goals = len(steering_goals)
    cols = 3
    rows = (num_goals + cols - 1) // cols  # Ceiling division to get the number of rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=capitalized_goals)

    # Get Plotly's default qualitative color palette
    default_colors = px.colors.qualitative.Plotly

    # Prepare colors for different methods using Plotly's default colors
    method_colors = {
        'CAA': default_colors[0],       # Typically blue
        'SAE': default_colors[1],            # Typically red/orange
        'SAE-TS': default_colors[2], # Typically green
    }

    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = idx // cols + 1
        col = idx % cols + 1
        goal_data = data_by_goal[goal]

        for method_data in goal_data:
            method = method_data['method']
            if method == 'RotationSteer':
                continue  # Skip RotationSteer
            if method == 'PinverseSteer':
                continue  # Skip PinverseSteer
            
            # Map the method names
            if method == 'ActSteer':
                method = 'CAA'
            elif method == 'OptimisedSteer':
                method = 'SAE-TS'
            
            scales = method_data['scales']
            product = method_data['product']
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=product,
                    mode='lines',
                    name=method,
                    line=dict(color=method_colors.get(method, default_colors[0])),  # Use Plotly default colors
                    showlegend=(row == 1 and col == 1)  # Show legend only in the first subplot
                ),
                row=row,
                col=col
            )

    # Update the layout of the figure
    fig.update_layout(
        height=300 * rows,  # Adjust the height as needed
        width=1200,         # Adjust the width as needed
        # title_text='Steering Analysis - Product vs Scale',
        legend_title='Method',
        showlegend=True,
        legend_font_size=14
    )

    # Update axis labels and fonts for all subplots
    fig.update_xaxes(title_text='Scale', row=rows, col=1)
    fig.update_xaxes(title_text='Scale', row=rows, col=2)
    fig.update_xaxes(title_text='Scale', row=rows, col=3)
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text='Behavioral*Coherence', row=row, col=1)

    # fig.update_yaxes(title_text='Coherence * Score', row=1, col=1)
    fig.update_annotations(font_size=18)

    # Save the figure as a JSON file
    if is_2b:
        output_filename = f"all_plots_2b{name}.png"
    else:
        output_filename = f"all_plots_9b{name}.png"
    pio.write_image(fig, output_filename)
    print(f"Figure saved as {output_filename}")

    # Optionally, still show the figure in the notebook
    fig.show()


def plot_specific_goals(data_path, is_2b=True, name=""):
    # Define the specific goals to plot
    goals_to_plot = ["London", "wedding"]

    # Load the graph data from the JSON file
    with open(data_path, 'r') as f:
        graph_data_list = json.load(f)

    # Filter the data for the specified goals and the 'OptimisedSteer' method
    graph_data_list = [
        data for data in graph_data_list
        if data['steering_goal_name'] in goals_to_plot and data['method'] == 'OptimisedSteer'
    ]

    # Get the unique steering goals (after filtering) and sort them case-insensitively
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list), key=str.lower)

    # Capitalize the steering goals for display
    capitalized_goals = [goal.capitalize() for goal in steering_goals]

    # Prepare a mapping from steering goal to its data
    data_by_goal = {goal: [] for goal in steering_goals}
    for data in graph_data_list:
        goal = data['steering_goal_name']
        data_by_goal[goal].append(data)

    # Create subplots: 2 columns (side by side), 1 row
    cols = 2
    rows = 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=capitalized_goals)

    # Line styles for different metrics
    metric_styles = {
        'product': {'dash': 'solid', 'width': 3},
        'avg_coherence': {'dash': 'dot', 'width': 2},
        'avg_score': {'dash': 'dash', 'width': 2},
    }

    # Get default Plotly colors
    default_colors = px.colors.qualitative.Plotly
    # Assign colors to metrics using default colors
    metric_colors = {
        'product': default_colors[2],       # Third color
        'avg_coherence': default_colors[1], # Second color
        'avg_score': default_colors[0],     # First color
    }

    legend_names = {
        'product': 'Behavioral*Coherence',
        'avg_coherence': 'Coherence',
        'avg_score': 'Behavioral'
    }

    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = 1
        col = idx + 1
        goal_data = data_by_goal[goal][0]  # Only one method ('OptimisedSteer') per goal

        # Limit scales to 180
        scales = [s for s in goal_data['scales'] if s <= 180]
        # Find indices where scales are <= 180
        valid_indices = [i for i, s in enumerate(goal_data['scales']) if s <= 180]

        # Get the corresponding metric values up to scale 180
        metrics = {
            'product': [goal_data['product'][i] for i in valid_indices],
            'avg_coherence': [goal_data['avg_coherence'][i] for i in valid_indices],
            'avg_score': [goal_data['avg_score'][i] for i in valid_indices],
        }

        for metric_name, metric_values in metrics.items():
            style = metric_styles.get(metric_name, {})
            color = metric_colors.get(metric_name, 'black')
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=metric_values,
                    mode='lines',
                    name=legend_names[metric_name] if idx == 0 else None,
                    line=dict(
                        color=color,
                        dash=style.get('dash', 'solid'),
                        width=style.get('width', 2),
                    ),
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )

    # Update the layout of the figure
    fig.update_layout(
        height=450,  # Reduced height for less tall plots
        width=1200,  # Adjusted width for side by side plots
        # title_text='Targeted Steering Metrics for London and Wedding',
        legend_title='Metric',
        showlegend=True,
        legend_font_size=16
    )

    # Update axis labels and fonts for all subplots
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text='Scale', row=1, col=c, range=[0, 180], tickfont=dict(size=14), title_font=dict(size=16))
        fig.update_yaxes(title_text='Score', row=1, col=c, range=[0, 1], tickfont=dict(size=14), title_font=dict(size=16))
    fig.update_annotations(font_size=18)

    # Save the figure as a JSON file
    if is_2b:
        output_filename = f"London_wedding_optimised_metrics_2b{name}.png"
    else:
        output_filename = f"London_wedding_optimised_metrics_9b{name}.png"
    pio.write_image(fig, output_filename)
    print(f"Figure saved as {output_filename}")

    # Optionally, show the figure in the notebook
    fig.show()

def plot_pareto_curves(data_path, is_2b=True, name=""):
    with open(data_path, 'r') as f:
        graph_data_list = json.load(f)

    # Get the unique steering goals and sort them case-insensitively
    steering_goals = sorted(set(data['steering_goal_name'] for data in graph_data_list), key=str.lower)

    # Capitalize the steering goals for display
    capitalized_goals = [goal.capitalize() for goal in steering_goals]

    # Prepare a mapping from steering goal to its data
    data_by_goal = {goal: [] for goal in steering_goals}
    for data in graph_data_list:
        data_by_goal[data['steering_goal_name']].append(data)

    # Create subplots
    num_goals = len(steering_goals)
    cols = 3
    rows = (num_goals + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=capitalized_goals)

    # Get Plotly's default qualitative color palette
    default_colors = px.colors.qualitative.Plotly

    # Prepare colors for different methods
    method_colors = {
        'CAA': default_colors[0],
        'SAE': default_colors[1],
        'SAE-TS': default_colors[2],
    }

    # Add data to each subplot
    for idx, goal in enumerate(steering_goals):
        row = idx // cols + 1
        col = idx % cols + 1
        goal_data_list = data_by_goal[goal]

        # Add "Good" label in top-right corner with grey color
        fig.add_annotation(
            x=0.9,
            y=0.9,
            text="Good",
            showarrow=False,
            xref=f'x{idx + 1}',
            yref=f'y{idx + 1}',
            font=dict(size=20, color='grey'),  # Added grey color
            xanchor='right',
            yanchor='top'
        )

        # Add "Bad" label in bottom-left corner with grey color
        fig.add_annotation(
            x=0.1,
            y=0.1,
            text="Bad",
            showarrow=False,
            xref=f'x{idx + 1}',
            yref=f'y{idx + 1}',
            font=dict(size=20, color='grey'),  # Added grey color
            xanchor='left',
            yanchor='bottom'
        )

        # Prepare data for each method
        method_data_dict = {}
        for method_data in goal_data_list:
            method = method_data['method']
            if method in ['RotationSteer', 'PinverseSteer']:
                continue

            # Map the method names
            method_name = {
                'ActSteer': 'CAA',
                'OptimisedSteer': 'SAE-TS',
                'DirectSteer': 'SAE'
            }.get(method, method)

            # Sort by scale to ensure correct arrow direction
            scales = method_data['scales']
            avg_coherence = method_data['avg_coherence']
            avg_score = method_data['avg_score']
            
            # Create sorted pairs of points based on scale
            points = list(zip(scales, avg_coherence, avg_score))
            points.sort(key=lambda x: x[0])  # Sort by scale
            
            method_data_dict[method_name] = points

        # Now plot for each method
        for method_name, points in method_data_dict.items():
            # Extract coherence and score values
            coherence_vals = [p[1] for p in points]
            score_vals = [p[2] for p in points]

            # Plot just the lines (no markers)
            fig.add_trace(
                go.Scatter(
                    x=coherence_vals,
                    y=score_vals,
                    mode='lines',
                    name=method_name,
                    line=dict(color=method_colors.get(method_name, default_colors[0])),
                    showlegend=(row == 1 and col == 1 and method_name not in [trace.name for trace in fig.data])
                ),
                row=row,
                col=col
            )

            # Add arrows between consecutive points
            for i in range(len(points) - 1):
                x0, y0 = coherence_vals[i], score_vals[i]
                x1, y1 = coherence_vals[i + 1], score_vals[i + 1]
                
                # Calculate arrow position (80% along the line)
                ax = x0 + 0.8 * (x1 - x0)
                ay = y0 + 0.8 * (y1 - y0)
                
                # Add arrow annotation with increased size
                fig.add_annotation(
                    x=ax,
                    y=ay,
                    ax=x0,
                    ay=y0,
                    xref=f'x{idx + 1}',
                    yref=f'y{idx + 1}',
                    axref=f'x{idx + 1}',
                    ayref=f'y{idx + 1}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=method_colors.get(method_name, default_colors[0])
                )

    # Update the layout
    fig.update_layout(
        height=300 * rows,
        width=1200,
        legend_title='Method',
        showlegend=True,
        legend_font_size=14
    )

    # Update axis labels and fonts
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(title_text='Coherence Score', row=r, col=c, range=[0, 1])
            fig.update_yaxes(title_text='Behavioral Score', row=r, col=c, range=[0, 1])

    fig.update_annotations(font_size=18)

    # Save the figure
    output_filename = f"pareto_curves_{('2b' if is_2b else '9b')}{name}.png"
    pio.write_image(fig, output_filename)
    print(f"Figure saved as {output_filename}")

    fig.show()
# %%
if __name__ == "__main__":
    name=""
    # name="_surprisingly"

    main(data_path=f"graph_data_all_methods_gemma-2-2b{name}.json", is_2b=True, name=name)
    plot_specific_goals(data_path=f"graph_data_all_methods_gemma-2-2b{name}.json", is_2b=True, name=name)
    plot_pareto_curves(data_path=f"graph_data_all_methods_gemma-2-2b{name}.json", is_2b=True, name=name)

    # main(data_path="graph_data_all_methods_gemma-2-9b.json", is_2b=False, name=name)
    # plot_specific_goals(data_path="graph_data_all_methods_gemma-2-9b.json", is_2b=False, name=name)
    # plot_pareto_curves(data_path="graph_data_all_methods_gemma-2-9b.json", is_2b=False, name=name)


# %%
