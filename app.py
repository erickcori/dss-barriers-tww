# Decision Support System: Systemic Barrier Analysis

# Importar librerías
import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table, callback
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import shap

# ========== DATOS ==========
data = {
    "Barrier": ['A1', 'A2', 'I1', 'I2', 'G1', 'G2', 'G4', 'G5', 'I5', 'F1', 'S1', 'S2'],
    "SHAP": [0.383, 0.383, 0.404, 0.387, 0.476, 0.454, 0.426, 0.694, 0.506, 0.481, 0.392, 0.421],
    "Out_Degree": [8.91, 4.00, 9.01, 11.33, 2.91, 12.25, 5.41, 16.93, 13.26, 11.73, 3.91, 3.55],
    "Betweenness": [0.114, 0.033, 0.039, 0.071, 0.026, 0.093, 0.005, 0.392, 0.026, 0.013, 0.133, 0.065]
}
df = pd.DataFrame(data)

# Matriz de impacto cruzado del paper
barriers = ['A1', 'A2', 'I1', 'I2', 'G1', 'G2', 'G4', 'G5', 'I5', 'F1', 'S1', 'S2']
cross_impact = np.array([
    [0.0, 2.46, 1.92, 0.0, 0.0, 0.0, 0.0, 2.15, 2.38, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.31, 1.69],
    [1.67, 2.17, 0.0, 0.0, 1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 2.17, 1.25],
    [2.5, 2.5, 2.58, 0.0, 2.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
    [1.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.33, 0.0],
    [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.58, 2.5, 2.25, 0.0, 1.92],
    [0.0, 0.0, 0.0, 1.83, 1.58, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [2.25, 2.25, 2.17, 2.17, 1.92, 2.08, 1.92, 0.0, 2.17, 0.0, 0.0, 0.0],
    [2.17, 2.42, 2.25, 0.0, 2.25, 0.0, 2.0, 2.17, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.67, 2.75, 2.5, 2.36, 0.0, 0.0, 0.0, 0.0, 0.0, 1.45, 0.0],
    [0.0, 0.0, 0.0, 1.73, 0.0, 0.0, 0.0, 2.18, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.82, 0.0, 0.0, 1.73, 0.0],
])
df_matrix = pd.DataFrame(cross_impact, index=barriers, columns=barriers)

# ========== FUNCIONES ANALÍTICAS ==========
def calculate_composite_score(df, weights=None):
    """Calculate composite score using min-max normalization"""
    if weights is None:
        weights = {'SHAP': 0.5, 'Out_Degree': 0.3, 'Betweenness': 0.2}
    scaler = MinMaxScaler()
    df_norm = df.copy()
    for col in ['SHAP', 'Out_Degree', 'Betweenness']:
        df_norm[col] = scaler.fit_transform(df[[col]])
    df['Composite_Score'] = (
        weights['SHAP'] * df_norm['SHAP'] +
        weights['Out_Degree'] * df_norm['Out_Degree'] +
        weights['Betweenness'] * df_norm['Betweenness']
    )
    return df

def calculate_shap_values(df, barrier_to_remove=None):
    """Recalculate SHAP values excluding the specified barrier"""
    if barrier_to_remove:
        if barrier_to_remove not in df['Barrier'].values:
            raise ValueError(f"Barrier '{barrier_to_remove}' not found in the list of barriers.")

	# Hacer una copia segura de cross_impact antes de modificarla
        cross_impact_no_barrier = cross_impact.copy()

	 # Encontrar índice de la barrera
        barrier_index = df.index[df['Barrier'] == barrier_to_remove].tolist()[0]


        # Eliminar fila y columna correspondiente a la barrera
        cross_impact_no_barrier = np.delete(cross_impact_no_barrier, barrier_index, axis=0)  # fila
        cross_impact_no_barrier = np.delete(cross_impact_no_barrier, barrier_index, axis=1)  # columna

        # Recalcular SHAP values sin la barrera
        shap_values_no_barrier = np.mean(cross_impact_no_barrier, axis=1)

        # Crear nuevo DataFrame sin la barrera
        df_no_barrier = df[df['Barrier'] != barrier_to_remove].copy()

        # Verificar que las longitudes coincidan antes de asignar
        if len(shap_values_no_barrier) != len(df_no_barrier):
            raise ValueError(f"Length mismatch: SHAP values ({len(shap_values_no_barrier)}) vs DataFrame ({len(df_no_barrier)})")

        # Asignar correctamente
        df_no_barrier['SHAP_No_Barrier'] = shap_values_no_barrier
        return df_no_barrier
    else:
        # Devolver SHAP originales
        shap_values = np.mean(cross_impact, axis=1)
        df['SHAP_Original'] = shap_values
        return df


def perform_clustering(df):
    """Perform PCA and K-Means clustering directly from cross-impact matrix"""
    X_scaled = StandardScaler().fit_transform(cross_impact)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    df['Cluster'] = clusters
    df['Cluster_Name'] = df['Cluster'].map({
        0: 'Structural Enablers',
        1: 'Peripheral Triggers',
        2: 'Context-Dependent Obstacles'
    })
    return df

def create_pca_plot(df):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig = go.Figure()
    shown_clusters = set()
    cluster_names = {0: 'Structural Enablers', 1: 'Peripheral Triggers', 2: 'Context-Dependent Obstacles'}
    for i, row in df.iterrows():
        cluster_id = int(row['Cluster'])
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        color = colors[cluster_id % len(colors)]
        fig.add_trace(go.Scatter(
            x=[row['PCA1']], y=[row['PCA2']],
            mode='markers+text',
            text=row['Barrier'],
            textposition='top center',
            marker=dict(size=10, color=color),
            name=cluster_name,
            legendgroup=cluster_name,
            showlegend=cluster_name not in shown_clusters
        ))
        shown_clusters.add(cluster_name)
        fig.update_layout(
        title="PCA Plot",
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False
    )
    return fig

def create_cluster_description(df):
    return [
        html.H6("", className="font-weight-bold"),
        html.P([
            html.Span("● ", style={'color': '#1f77b4'}),
            html.Strong("Cluster 0: "),
            "Structural Enablers"
        ]),
        html.P([
            html.Span("● ", style={'color': '#ff7f0e'}),
            html.Strong("Cluster 1: "),
            "Peripheral Triggers"
        ]),
        html.P([
            html.Span("● ", style={'color': '#2ca02c'}),
            html.Strong("Cluster 2: "),
            "Context-Dependent Obstacles"
        ])
    ]


def simulate_removal(df, barrier_to_remove):
    """Simulate barrier removal"""
    df_sim = df.copy()
    if barrier_to_remove:
        df_sim.loc[df_sim['Barrier'] == barrier_to_remove, ['SHAP', 'Out_Degree', 'Betweenness']] = 0
    return df_sim

def calculate_shap_values(df, barrier_to_remove=None):
    """Recalculate SHAP values excluding the specified barrier"""
    if barrier_to_remove:
        if barrier_to_remove not in df['Barrier'].values:
            raise ValueError(f"Barrier '{barrier_to_remove}' not found in the list of barriers.")
        barrier_index = df.index[df['Barrier'] == barrier_to_remove].tolist()[0]
        cross_impact_no_barrier = np.delete(np.delete(cross_impact, barrier_index, axis=0), barrier_index, axis=1)
        shap_no_barrier = np.mean(cross_impact_no_barrier, axis=1)
        df_no_barrier = df[df['Barrier'] != barrier_to_remove].copy()
        df_no_barrier['SHAP_No_Barrier'] = shap_no_barrier
        return df_no_barrier
    else:
        shap_values = np.mean(cross_impact, axis=1)
        df['SHAP_Original'] = shap_values
        return df

# Inicialización inicial
df = calculate_composite_score(df)
df = perform_clustering(df)

# ========== APLICACIÓN DASH ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = dbc.Container([
    # Encabezado
    dbc.Row([
        dbc.Col(html.H1("Decision Support System: Systemic Barrier Analysis", className="text-center my-4"), width=12)
    ]),
    # Descripción
    dbc.Row([
        dbc.Col(html.P("""Interactive dashboard for barrier prioritization based on systemic influence (SHAP), network centrality, and cluster analysis.""",
                       className="text-center text-muted mb-4"), width=12)
    ]),
    # Controles
    dbc.Row([
        dbc.Col([
            html.H5("Weight Configuration", className="mb-3"),
            dbc.Label("SHAP Weight (Systemic Influence):", className="font-weight-bold"),
            dcc.Slider(id='shap-weight', min=0, max=1, step=0.1, value=0.5,
                       marks={i / 10: str(i / 10) for i in range(0, 11)}),
            dbc.Label("Out-Degree Weight (Network Centrality):", className="font-weight-bold mt-3"),
            dcc.Slider(id='outdegree-weight', min=0, max=1, step=0.1, value=0.3,
                       marks={i / 10: str(i / 10) for i in range(0, 11)}),
            dbc.Label("Betweenness Weight (Mediation Role):", className="font-weight-bold mt-3"),
            dcc.Slider(id='betweenness-weight', min=0, max=1, step=0.1, value=0.2,
                       marks={i / 10: str(i / 10) for i in range(0, 11)})
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
                dbc.Col(dcc.Graph(id='pca-plot'), width=8),
                dbc.Col([
                    html.H5("Cluster Description", className="mb-3"),
                    html.Div(id='cluster-description', className="p-3 bg-light rounded")
                ], width=4)
            ])
        ], label="Typology Explorer"),
        dbc.Tab([
            dbc.Row([
                dbc.Col(dcc.Graph(id='impact-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        id='impact-table',
                        columns=[
                            {"name": "Barrier", "id": "Barrier"},
                            {"name": "Original Score", "id": "Original", "format": {"specifier": ".3f"}},
                            {"name": "New Score", "id": "New Score", "format": {"specifier": ".3f"}},
                            {"name": "Change", "id": "Change", "format": {"specifier": ".4f"}}
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
                    ), width=12)
            ])
        ], label="Impact Simulator")
    ]),

    # Notas al pie
    dbc.Row([
        dbc.Col(
            html.P("""
                Note: The composite score combines systemic influence (SHAP), centrality (Out-Degree),
                and mediation role (Betweenness) using the specified weights. Clusters were identified
                using K-means (k=3) on the PCA space from the cross-impact matrix.
            """, className="small text-muted mt-4"),
            width=12
        )
    ])
], fluid=True, style={"color": "#000"})

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
    total = shap_w + out_w + bet_w
    weights = {
        'SHAP': shap_w / total,
        'Out_Degree': out_w / total,
        'Betweenness': bet_w / total
    }
    
    df_sim = simulate_removal(df, barrier_removed)
    df_sim = calculate_composite_score(df_sim, weights)

    if clusters:
        if isinstance(clusters, str):
            clusters = [clusters]
        df_sim = df_sim[df_sim['Cluster'].isin(clusters)]
    
    df_sorted = df_sim.sort_values(by=sort_by, ascending=False)

    impact_fig = go.Figure()
    impact_table = []
    sim_summary = html.P("Select a barrier to simulate its impact on the system.")
    if barrier_removed:
        df_updated = calculate_composite_score(df.copy(), weights)

        df_shap = calculate_shap_values(df_updated, barrier_removed)
        original_shap = df_updated['SHAP']

        try:
            shap_no_barrier = df_shap['SHAP_No_Barrier']
            delta = original_shap - shap_no_barrier
            epsilon = 1e-8
            percent_change = ((delta / (original_shap + epsilon))*100).clip(-100, 100)  # entre -1000% y +1000%
        except KeyError:
            percent_change = pd.Series([0] * len(df))

        # Crear tabla base
        df_temp = df[['Barrier']].copy()
        df_temp['Original'] = df['Composite_Score']

        print("\n--- df_shap ---")
        print(df_shap[['Barrier', 'SHAP_No_Barrier']])

        # Merge con los SHAP recalculados
        df_temp = df_temp.merge(df_shap[['Barrier', 'SHAP_No_Barrier']], on='Barrier', how='left')
        

        # renombrar para que quede elegante
        df_temp.rename(columns={'SHAP_No_Barrier': 'New Score'}, inplace=True)

        # opcional: si quieres reemplazar NaN por cero o por guión

        df_temp['New Score'] = df_temp['New Score'].fillna(0)  # o .fillna('-') si prefieres

        print("\n--- df_temp after merge ---")
        print(df_temp.head(12))




        # ====================== CÁLCULO CLARO Y ROBUSTO DE CHANGE ======================
        # Validación previa de columnas necesarias
        required_columns = ['Original', 'New Score']
        for col in required_columns:
            if col not in df_temp.columns:
                raise KeyError(f"Column '{col}' is missing in df_temp")

        # ====================== CAMBIO ABSOLUTO EN LUGAR DE PORCENTUAL ======================
        df_temp['Change'] = (df_temp['New Score'] - df_temp['Original']).round(4)

        # Clipping para evitar valores fuera de escala razonable
        df_temp['Change'] = df_temp['Change'].clip(-100, 1000)

        # Redondeo para presentación
        df_temp['Change'] = df_temp['Change'].round(2)

        # Ordenar para mostrar los cambios más relevantes
        df_temp = df_temp.sort_values(by='Change', key=lambda x: abs(x), ascending=False)


        # Gráfico de barras horizontales
        if 'percent_change' in locals():
            sorted_idx = np.argsort(abs(percent_change))[::-1]
            barriers_sorted = df['Barrier'].iloc[sorted_idx].values
            percent_change_sorted = percent_change.iloc[sorted_idx].values
        else:
            barriers_sorted = df['Barrier'].values
            percent_change_sorted = pd.Series([0]*len(df))

        impact_colors = ['#3498db' if x < 0 else '#e74c3c' for x in percent_change_sorted]

        impact_fig = go.Figure()

        impact_fig.add_trace(go.Bar(
            x=percent_change_sorted,
            y=barriers_sorted,
            orientation='h',
            marker_color=impact_colors,
            width=0.7,
            text=[f'{v:.1f}%' for v in percent_change_sorted],
            textposition='outside',
            textfont=dict(size=10, color='black'),
            cliponaxis=False,  # Para que no se corte el texto
            hovertemplate="<b>Barrier:</b> %{y}<br><b>Change:</b> %{x:.1f}%<extra></extra>",
            showlegend=False
        ))
        impact_fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
        impact_fig.update_layout(
            title=f"Impact of Removing {barrier_removed} on Other Barriers",
            xaxis_title="Percentage Change in SHAP Value (%)",
            yaxis_title="Barriers",
            xaxis_tickformat=".1f",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=10),
            margin=dict(l=80, r=80, t=60, b=80),
            height=500
        )
        max_abs = max(abs(percent_change.min()), abs(percent_change.max()), 20)
        impact_fig.update_xaxes(
            range=[-1.2 * max_abs, 1.2 * max_abs],
            dtick=20,
            tickformat=".1f",
            title_text="Percentage Change in SHAP Value (%)",
            showgrid=True,
            gridcolor='lightgray'
        )

        impact_fig.update_yaxes(tickmode='linear', dtick=1, gridwidth=1, gridcolor='lightgray')

        # Asignar tabla
        impact_table = df_temp[['Barrier', 'Original', 'New Score', 'Change']].to_dict('records')

        # Resumen de simulación
        try:
            removed_score = df_updated[df_updated['Barrier'] == barrier_removed]['Composite_Score'].values[0]
        except IndexError:
            removed_score = 0

        max_increase = percent_change.max() if 'percent_change' in locals() and not np.isnan(percent_change.max()) else 0
        max_decrease = percent_change.min() if 'percent_change' in locals() and not np.isnan(percent_change.min()) else 0

        sim_summary = [
            html.H6(f"Simulation: Removal of {barrier_removed}", className="font-weight-bold"),
            html.P(f"Original score: {removed_score:.3f}"),
            html.P(f"Highest increase: {max_increase:.1f}%"),
            html.P(f"Largest decrease: {max_decrease:.1f}%")
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

    pca_fig = create_pca_plot(df_sim)
    cluster_desc = create_cluster_description(df_sim)

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
    app.run_server(debug=False)
