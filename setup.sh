#!/bin/bash

# Script to set up Prometheus and Grafana monitoring for Ella-Core
# Run this from your ~/ella directory

echo "🔧 Setting up monitoring stack for Ella-Core..."

# Create required directories
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards

echo "✅ Created grafana provisioning directories"

# Create datasources config directory structure
cat > grafana/provisioning/datasources/prometheus.yaml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

echo "✅ Created Prometheus datasource config"

# Create dashboard provisioning config
cat > grafana/provisioning/dashboards/dashboard.yaml << 'EOF'
apiVersion: 1

providers:
  - name: 'Ella-Core Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

echo "✅ Created dashboard provisioning config"

# Create the dashboard JSON file
cat > grafana/provisioning/dashboards/ella-core-dashboard.json << 'DASHBOARD'
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {"tooltip": false, "viz": false, "legend": false},
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {"type": "linear"},
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {"group": "A", "mode": "none"},
            "thresholdsStyle": {"mode": "off"}
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}, {"color": "red", "value": 80}]
          }
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "id": 2,
      "options": {
        "legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": true},
        "tooltip": {"mode": "single", "sort": "none"}
      },
      "pluginVersion": "10.0.0",
      "targets": [{"expr": "up{job=\"ella-core\"}", "refId": "A"}],
      "title": "Ella-Core Service Status",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}, {"color": "red", "value": 0}]
          }
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "id": 3,
      "options": {
        "colorMode": "background",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {
          "values": false,
          "fields": "",
          "calcs": ["lastNotNull"]
        },
        "text": {},
        "textMode": "auto"
      },
      "pluginVersion": "10.0.0",
      "targets": [{"expr": "up{job=\"ella-core\"}", "refId": "A"}],
      "title": "Ella-Core Status (1=Up, 0=Down)",
      "type": "stat"
    }
  ],
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["ella-core", "5g", "qos"],
  "templating": {"list": []},
  "time": {"from": "now-6h", "to": "now"},
  "timepicker": {},
  "timezone": "",
  "title": "Ella-Core 5G QoS Monitoring",
  "uid": "ella-core-qos",
  "version": 0,
  "weekStart": ""
}
DASHBOARD

echo "✅ Created Ella-Core dashboard JSON"

echo ""
echo "📋 Setup complete! Next steps:"
echo ""
echo "1. Replace your current docker-compose.yaml with the new one containing Prometheus and Grafana"
echo "2. Run: docker-compose down"
echo "3. Run: docker-compose up -d"
echo "4. Check services:"
echo "   - Ella-Core API: http://192.168.56.11:5002"
echo "   - Prometheus: http://192.168.56.11:9090"
echo "   - Grafana: http://192.168.56.11:3000 (admin/admin)"
echo ""
echo "5. In Grafana, import the dashboard JSON file from the outputs folder"
echo "6. Check Prometheus targets at: http://192.168.56.11:9090/targets"
echo ""