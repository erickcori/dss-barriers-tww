# Decision Support System: Systemic Barrier Analysis
# Install required dependencies

# Importar librerías
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 👈 Añade StandardScaler aquí
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dash import Dash, dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc

# ========== DATOS ==========
data = {
    "Barrier": ['A1', 'A2', 'F1', 'G1', 'G2', 'G4', 'G5', 'I1', 'I2', 'I5', 'S1', 'S2'],
    "SHAP": [0.383, 0.383, 0.481, 0.476, 0.454, 0.426, 0.694, 0.404, 0.387, 0.506, 0.392, 0.421],
    "Out_Degree": [14.53, 4.00, 15.25, 5.41, 13.26, 10.00, 10.91, 14.23, 13.87, 13.01, 15.26, 8.90],
    "Betweenness": [0.0000, 0.0000, 0.0000, 0.0000, 0.0556, 0.0556, 0.0000, 0.0556, 0.1111, 0.0556, 0.0000, 0.1111]
}
df = pd.DataFrame(data)

# ========== FUNCIONES ANALÍTICAS ==========
def calculate_composite_score(df, weights=None):
    """Calculate composite score using min-max normalization"""
    if weights is None:
        weights = {'SHAP': 0.5, 'Out_Degree': 0.3, 'Betweenness': 0.2}

    # NNormalization
    scaler = MinMaxScaler()
    df_norm = df.copy()
    for col in ['SHAP', 'Out_Degree', 'Betweenness']:
        df_norm[col] = scaler.fit_transform(df[[col]])

    # Composite score
    df['Composite_Score'] = (weights['SHAP'] * df_norm['SHAP'] +
                           weights['Out_Degree'] * df_norm['Out_Degree'] +
                           weights['Betweenness'] * df_norm['Betweenness'])
    return df

def perform_clustering(df):
    """Perform PCA and K-means clustering"""
    # Estandarización
    X = df[['SHAP', 'Out_Degree', 'Betweenness']]
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].map({0: 'Structural Enablers',
                                     1: 'Peripheral Triggers',
                                     2: 'Context-Dependent Obstacles'})
    return df

def simulate_removal(df, barrier_to_remove):
    """Simulate barrier removal"""
    df_sim = df.copy()
    if barrier_to_remove:
        df_sim.loc[df_sim['Barrier'] == barrier_to_remove, ['SHAP', 'Out_Degree', 'Betweenness']] = 0
    return df_sim

# Initial processing
df = calculate_composite_score(df)
df = perform_clustering(df)

# ========== APLICACIÓN DASH ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = dbc.Container([
    # Encabezado
    dbc.Row([
        dbc.Col(html.H1("Decision Support System: Systemic Barrier Analysis",
                       className="text-center my-4"),
               width=12)
    ]),

    # Descripción
    dbc.Row([
        dbc.Col(html.P("""Interactive dashboard for barrier prioritization based on systemic influence (SHAP), network centrality, and cluster analysis.""",
                className="text-center text-muted mb-4"),
               width=12)
    ]),

    # Controles
    dbc.Row([
        dbc.Col([
            html.H5("Weight Configuration", className="mb-3"),
            dbc.Label("SHAP Weight (Systemic Influence):", className="font-weight-bold"),
            dcc.Slider(id='shap-weight', min=0, max=1, step=0.1, value=0.5,
                      marks={i/10: str(i/10) for i in range(0, 11)}),

            dbc.Label("Out-Degree Weight (Network Centrality):", className="font-weight-bold mt-3"),
            dcc.Slider(id='outdegree-weight', min=0, max=1, step=0.1, value=0.3,
                      marks={i/10: str(i/10) for i in range(0, 11)}),

            dbc.Label("Betweenness Weight (Mediation Role):", className="font-weight-bold mt-3"),
            dcc.Slider(id='betweenness-weight', min=0, max=1, step=0.1, value=0.2,
                      marks={i/10: str(i/10) for i in range(0, 11)}),
        ], md=4),

        dbc.Col([
            html.H5("What-If Simulation", className="mb-3"),
            dbc.Label("Select a barrier to simulate its removal:", className="font-weight-bold"),
            dcc.Dropdown(
                id='barrier-dropdown',
                options=[{'label': b, 'value': b} for b in df['Barrier']],
                value=None,
                placeholder="Select a barrier...",
                clearable=True
            ),

            html.Div(id='simulation-summary', className="mt-3 p-3 bg-light rounded")
        ], md=4),

        dbc.Col([
            html.H5("Filters", className="mb-3"),
            dbc.Label("Filter by Cluster:", className="font-weight-bold"),
            dcc.Dropdown(
                id='cluster-filter',
                options=[{'label': c, 'value': c} for c in df['Cluster'].unique()],
                value=None,
                placeholder="All clusters",
                multi=True
            ),

            dbc.Label("Sort by:", className="font-weight-bold mt-3"),
            dcc.Dropdown(
                id='sort-by',
                options=[
                    {'label': 'Composite Score', 'value': 'Composite_Score'},
                    {'label': 'SHAP', 'value': 'SHAP'},
                    {'label': 'Out-Degree', 'value': 'Out_Degree'},
                    {'label': 'Betweenness', 'value': 'Betweenness'}
                ],
                value='Composite_Score'
            )
        ], md=4)
    ], className="mb-4"),

    # Pestañas
    dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        id='ranking-table',
                        columns=[
                            {"name": "Barrier", "id": "Barrier"},
                            {"name": "SHAP", "id": "SHAP", "format": {"specifier": ".3f"}},
                            {"name": "Out-Degree", "id": "Out_Degree", "format": {"specifier": ".2f"}},
                            {"name": "Betweenness", "id": "Betweenness", "format": {"specifier": ".4f"}},
                            {"name": "Composite Score", "id": "Composite_Score", "format": {"specifier": ".3f"}},
                            {"name": "Cluster", "id": "Cluster"}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '8px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            },
                            {
                                'if': {'column_id': 'Composite_Score', 'filter_query': '{Composite_Score} > 0.7'},
                                'backgroundColor': '#d4edda',
                                'fontWeight': 'bold'
                            }
                        ],
                        sort_action="native",
                        sort_mode="single",
                        filter_action="native",
                        page_size=12
                    ),
                    width=12
                )
            ])
        ], label="Barrier Ranking"),

        dbc.Tab([
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='pca-plot'),
                    width=8
                ),
                dbc.Col([
                    html.H5("Cluster Description", className="mb-3"),
                    html.Div(id='cluster-description', className="p-3 bg-light rounded")
                ], width=4)
            ])
        ], label="Typology Explorer"),

        dbc.Tab([
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='impact-plot'),
                    width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        id='impact-table',
                        columns=[
                            {"name": "Barrier", "id": "Barrier"},
                            {"name": "Original Score", "id": "Original", "format": {"specifier": ".3f"}},
                            {"name": "New Score", "id": "New", "format": {"specifier": ".3f"}},
                            {"name": "Change", "id": "Change", "format": {"specifier": ".2%"}}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '8px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'Change', 'filter_query': '{Change} > 0'},
                                'color': 'green'
                            },
                            {
                                'if': {'column_id': 'Change', 'filter_query': '{Change} < 0'},
                                'color': 'red'
                            }
                        ]
                    ),
                    width=12
                )
            ])
        ], label="Impact Simulator")
    ]),

    # Notas al pie
    dbc.Row([
        dbc.Col(
            html.P("""
                 Note: The composite score combines systemic influence (SHAP), centrality (Out-Degree),
                and mediation role (Betweenness) using the specified weights. Clusters were identified
                using K-means (k=3) on standardized metrics.
            """, className="small text-muted mt-4"),
            width=12
        )
    ])
], fluid=True)

# ========== CALLBACKS ==========
@callback(
    Output('ranking-table', 'data'),
    Output('pca-plot', 'figure'),
    Output('impact-plot', 'figure'),
    Output('impact-table', 'data'),
    Output('simulation-summary', 'children'),
    Output('cluster-description', 'children'),
    Input('shap-weight', 'value'),
    Input('outdegree-weight', 'value'),
    Input('betweenness-weight', 'value'),
    Input('barrier-dropdown', 'value'),
    Input('cluster-filter', 'value'),
    Input('sort-by', 'value')
)
def update_dashboard(shap_w, out_w, bet_w, barrier_removed, clusters, sort_by):
    # Calcular pesos y normalizar
    total = shap_w + out_w + bet_w
    weights = {
        'SHAP': shap_w/total,
        'Out_Degree': out_w/total,
        'Betweenness': bet_w/total
    }

    # Simular eliminación si se especificó
    df_sim = simulate_removal(df, barrier_removed)
    df_sim = calculate_composite_score(df_sim, weights)

    # Filtrar por clusters si se especificó
    if clusters:
        if isinstance(clusters, str):
            clusters = [clusters]
        df_sim = df_sim[df_sim['Cluster'].isin(clusters)]

    # Ordenar
    df_sorted = df_sim.sort_values(by=sort_by, ascending=False)

    # Crear gráfico PCA
    pca_fig = px.scatter(
        df_sim, x='PCA1', y='PCA2', color='Cluster',
        hover_data=['Barrier', 'SHAP', 'Out_Degree', 'Betweenness', 'Composite_Score'],
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Barrier Typology Space"
    )
    pca_fig.update_layout(
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(255,255,255,1)',
        legend_title="Cluster"
    )

    # Gráfico de impacto si se simuló eliminación
    if barrier_removed:
        # Calcular cambio porcentual
        df_impact = df.copy()
        df_impact = calculate_composite_score(df_impact, weights)
        df_impact = df_impact.merge(
            df_sim[['Barrier', 'Composite_Score']],
            on='Barrier',
            suffixes=('_original', '_new')
        )
        df_impact['Change'] = (df_impact['Composite_Score_new'] - df_impact['Composite_Score_original']) / df_impact['Composite_Score_original']

        # Ordenar por magnitud de cambio
        df_impact = df_impact.sort_values(by='Change', key=abs, ascending=False)

        # Gráfico de barras
        impact_fig = px.bar(
            df_impact, x='Barrier', y='Change',
            color='Change', color_continuous_scale='RdBu',
            title=f"Variation in Composite Score Due to Removal of  {barrier_removed}"
        )
        impact_fig.update_layout(yaxis_tickformat=".0%")


        # Tabla de impacto
        impact_table = df_impact.rename(columns={
            'Composite_Score_original': 'Original',
            'Composite_Score_new': 'New'
        }).to_dict('records')

        # Resumen de simulación
        removed_score = df[df['Barrier'] == barrier_removed]['Composite_Score'].values[0]
        max_increase = df_impact['Change'].max()
        max_decrease = df_impact['Change'].min()

        sim_summary = [
    html.H6(f"Simulation: Removal of {barrier_removed}", className="font-weight-bold"),
    html.P(f"Original score: {removed_score:.3f}"),
    html.P(f"Highest increase: {max_increase:.1%}"),
    html.P(f"Largest decrease: {max_decrease:.1%}")
]
    else:
        impact_fig = go.Figure()
        impact_fig.update_layout(
            title="Select a barrier to simulate its removal",
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='rgba(240,240,240,0.8)'
        )
        impact_table = []
        sim_summary = html.P("Select a barrier to simulate its impact on the system.")

    # Descripción de clusters
    cluster_desc = [
        html.H6("Cluster Description", className="font-weight-bold"),
        html.P("Structural Enablers: Barriers deeply embedded in the network with high systemic influence.", className="mt-2"),
        html.P("Peripheral Triggers: Barriers with localized influence but potential for systemic ripple effects.", className="mt-2"),
        html.P("Context-Dependent Obstacles: Barriers whose relevance varies depending on scenario assumptions.", className="mt-2")
    ]

    return (
        df_sorted.to_dict('records'),
        pca_fig,
        impact_fig,
        impact_table,
        sim_summary,
        cluster_desc
    )


# ========== EJECUCIÓN ==========
if __name__ == '__main__':
    app.run(debug=False)
