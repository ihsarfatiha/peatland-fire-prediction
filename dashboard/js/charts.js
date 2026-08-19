/**
 * Peatland Fire Dashboard - Charts Module
 * Chart.js visualizations for peatland and climate data
 */

// Chart.js global defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

let provinceChart, islandChart, dryDaysChart, rainfallChart, lstChart, ndviChart;

// Color palette
const CHART_COLORS = {
  fire: ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16'],
  islands: {
    'SUMATERA': '#ef4444',
    'KALIMANTAN': '#f97316', 
    'PAPUA': '#3b82f6',
    'SULAWESI': '#8b5cf6',
    'NUSA TENGGARA': '#06b6d4',
    'JAWA': '#22c55e',
    'MALUKU': '#ec4899'
  },
  risk: {
    critical: '#dc2626',
    high: '#f97316',
    moderate: '#f59e0b',
    low: '#22c55e'
  }
};

function initCharts() {
  createProvinceChart();
  createIslandDoughnutChart();
  createDryDaysChart();
  createRainfallChart();
  createLstChart();
  createNdviChart();
}

// --- Top 10 Provinces Bar Chart ---
function createProvinceChart() {
  const ctx = document.getElementById('province-chart').getContext('2d');
  const top10 = PEAT_PROVINCES.slice(0, 10);
  
  const gradient = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
  gradient.addColorStop(0, '#ef4444');
  gradient.addColorStop(0.5, '#f97316');
  gradient.addColorStop(1, '#f59e0b');
  
  provinceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top10.map(p => p.region),
      datasets: [{
        label: 'Luas Lahan Gambut (Ha)',
        data: top10.map(p => Math.round(p.area)),
        backgroundColor: top10.map((_, i) => {
          const alpha = 1 - (i * 0.07);
          return `rgba(249, 115, 22, ${alpha})`;
        }),
        borderColor: 'rgba(249, 115, 22, 0.8)',
        borderWidth: 1,
        borderRadius: 4,
        barPercentage: 0.7
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          padding: 12,
          titleFont: { weight: '600' },
          callbacks: {
            label: function(ctx) {
              return `${(ctx.parsed.x).toLocaleString()} hektar`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            callback: v => (v / 1000000).toFixed(1) + 'M',
            font: { size: 11 }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { size: 11 },
            callback: function(value) {
              const label = this.getLabelForValue(value);
              return label.length > 18 ? label.substring(0, 16) + '…' : label;
            }
          }
        }
      }
    }
  });
}

// --- Island Doughnut Chart ---
function createIslandDoughnutChart() {
  const ctx = document.getElementById('island-chart').getContext('2d');
  
  const islands = Object.entries(PEAT_BY_ISLAND)
    .sort((a, b) => b[1] - a[1]);
  
  islandChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: islands.map(([name]) => name),
      datasets: [{
        data: islands.map(([, area]) => Math.round(area)),
        backgroundColor: islands.map(([name]) => CHART_COLORS.islands[name] || '#64748b'),
        borderColor: '#0a0e17',
        borderWidth: 2,
        hoverOffset: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 16,
            usePointStyle: true,
            pointStyle: 'circle',
            font: { size: 11, weight: '500' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: function(ctx) {
              const pct = ((ctx.parsed / TOTAL_PEATLAND_HECTARES) * 100).toFixed(1);
              return ` ${ctx.label}: ${(ctx.parsed).toLocaleString()} ha (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

// --- Dry Days Distribution ---
function createDryDaysChart() {
  const ctx = document.getElementById('dry-days-chart').getContext('2d');
  
  const bins = RIAU_STATS.dryDaysBins;
  const sortedBins = Object.keys(bins).map(Number).sort((a, b) => a - b);
  
  dryDaysChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sortedBins.map(b => `${b}-${b+9}`),
      datasets: [{
        label: 'Jumlah Titik',
        data: sortedBins.map(b => bins[b]),
        backgroundColor: sortedBins.map(b => {
          if (b >= 90) return 'rgba(220, 38, 38, 0.7)';
          if (b >= 80) return 'rgba(249, 115, 22, 0.7)';
          if (b >= 70) return 'rgba(245, 158, 11, 0.7)';
          return 'rgba(34, 197, 94, 0.7)';
        }),
        borderColor: sortedBins.map(b => {
          if (b >= 90) return '#dc2626';
          if (b >= 80) return '#f97316';
          if (b >= 70) return '#f59e0b';
          return '#22c55e';
        }),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            title: ctx => `${ctx[0].label} hari kering`,
            label: ctx => `${ctx.parsed.y} titik pengamatan`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          title: { display: true, text: 'Hari Kering Berturut-turut', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Jumlah Titik', font: { size: 11 } }
        }
      }
    }
  });
}

// --- Rainfall Distribution ---
function createRainfallChart() {
  const ctx = document.getElementById('rainfall-chart').getContext('2d');
  
  const bins = RIAU_STATS.rainfallBins;
  const sortedBins = Object.keys(bins).map(Number).sort((a, b) => a - b);
  
  rainfallChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sortedBins.map(b => `${b}-${b+1}`),
      datasets: [{
        label: 'Jumlah Titik',
        data: sortedBins.map(b => bins[b]),
        backgroundColor: sortedBins.map(b => {
          if (b < 3) return 'rgba(220, 38, 38, 0.7)';
          if (b < 4) return 'rgba(249, 115, 22, 0.7)';
          if (b < 5) return 'rgba(245, 158, 11, 0.7)';
          return 'rgba(59, 130, 246, 0.7)';
        }),
        borderColor: sortedBins.map(b => {
          if (b < 3) return '#dc2626';
          if (b < 4) return '#f97316';
          if (b < 5) return '#f59e0b';
          return '#3b82f6';
        }),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            title: ctx => `${ctx[0].label} mm/hari`,
            label: ctx => `${ctx.parsed.y} titik pengamatan`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          title: { display: true, text: 'Curah Hujan Rata-rata (mm/hari)', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Jumlah Titik', font: { size: 11 } }
        }
      }
    }
  });
}

// --- LST Scatter Plot ---
function createLstChart() {
  const ctx = document.getElementById('lst-chart').getContext('2d');
  
  const scatterData = RIAU_POINTS.map(p => {
    const risk = getRiskLevel(p[2], p[4]);
    return {
      x: p[2],  // dry days
      y: p[5],  // mean LST
      risk: risk.level,
      color: risk.color
    };
  });
  
  lstChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Kritis',
          data: scatterData.filter(d => d.risk === 'critical').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(220, 38, 38, 0.6)',
          borderColor: '#dc2626',
          pointRadius: 3
        },
        {
          label: 'Tinggi',
          data: scatterData.filter(d => d.risk === 'high').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(249, 115, 22, 0.6)',
          borderColor: '#f97316',
          pointRadius: 3
        },
        {
          label: 'Sedang',
          data: scatterData.filter(d => d.risk === 'moderate').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(245, 158, 11, 0.6)',
          borderColor: '#f59e0b',
          pointRadius: 3
        },
        {
          label: 'Rendah',
          data: scatterData.filter(d => d.risk === 'low').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: '#22c55e',
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, pointStyle: 'circle', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: ctx => `Hari Kering: ${ctx.parsed.x}, LST: ${ctx.parsed.y.toFixed(1)}°C`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Hari Kering Berturut-turut', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Suhu Permukaan Rata-rata (°C)', font: { size: 11 } }
        }
      }
    }
  });
}

// --- NDVI vs Rainfall Scatter ---
function createNdviChart() {
  const ctx = document.getElementById('ndvi-chart').getContext('2d');
  
  const scatterData = RIAU_POINTS.map(p => {
    const risk = getRiskLevel(p[2], p[4]);
    return {
      x: p[4],  // mean daily rainfall
      y: p[6],  // min NDVI
      risk: risk.level,
      color: risk.color
    };
  });
  
  ndviChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Kritis',
          data: scatterData.filter(d => d.risk === 'critical').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(220, 38, 38, 0.6)',
          borderColor: '#dc2626',
          pointRadius: 3
        },
        {
          label: 'Tinggi',
          data: scatterData.filter(d => d.risk === 'high').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(249, 115, 22, 0.6)',
          borderColor: '#f97316',
          pointRadius: 3
        },
        {
          label: 'Sedang',
          data: scatterData.filter(d => d.risk === 'moderate').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(245, 158, 11, 0.6)',
          borderColor: '#f59e0b',
          pointRadius: 3
        },
        {
          label: 'Rendah',
          data: scatterData.filter(d => d.risk === 'low').map(d => ({ x: d.x, y: d.y })),
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: '#22c55e',
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, pointStyle: 'circle', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: ctx => `Curah Hujan: ${ctx.parsed.x.toFixed(2)} mm/hari, NDVI: ${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Curah Hujan Rata-rata (mm/hari)', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'NDVI Minimum', font: { size: 11 } }
        }
      }
    }
  });
}
