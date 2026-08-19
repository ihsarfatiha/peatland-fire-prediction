/**
 * Peatland Fire Dashboard - Map Module
 * Leaflet.js interactive map for Riau fire risk points
 */

let map;
let markersLayer;

function initMap() {
  // Initialize map centered on Riau province
  map = L.map('fire-risk-map', {
    zoomControl: true,
    scrollWheelZoom: true,
    attributionControl: true
  }).setView([0.5, 101.5], 8);

  // Dark tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);

  // Add markers
  markersLayer = L.layerGroup().addTo(map);
  addFireRiskMarkers('all');
}

function addFireRiskMarkers(filter) {
  markersLayer.clearLayers();
  
  RIAU_POINTS.forEach(point => {
    const [lat, lon, dryDays, lstMax, meanRain, meanLst, minNdvi, totalRain] = point;
    const risk = getRiskLevel(dryDays, meanRain);
    
    // Filter logic
    if (filter !== 'all' && risk.level !== filter) return;
    
    // Create circle marker
    const marker = L.circleMarker([lat, lon], {
      radius: 5,
      fillColor: risk.color,
      color: risk.color,
      weight: 1,
      opacity: 0.8,
      fillOpacity: 0.6
    });
    
    // Popup content
    const popupContent = `
      <div class="popup-title">
        <span style="color:${risk.color}">●</span>
        Risiko ${risk.label}
      </div>
      <div class="popup-row">
        <span class="popup-label">Koordinat</span>
        <span class="popup-value">${lat.toFixed(3)}, ${lon.toFixed(3)}</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">Hari Kering</span>
        <span class="popup-value">${dryDays} hari</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">LST Maks</span>
        <span class="popup-value">${lstMax.toFixed(1)}°C</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">Curah Hujan</span>
        <span class="popup-value">${meanRain.toFixed(2)} mm/hr</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">LST Rata-rata</span>
        <span class="popup-value">${meanLst.toFixed(1)}°C</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">NDVI Min</span>
        <span class="popup-value">${minNdvi.toFixed(3)}</span>
      </div>
      <div class="popup-row">
        <span class="popup-label">Total Hujan Musim</span>
        <span class="popup-value">${totalRain.toFixed(0)} mm</span>
      </div>
    `;
    
    marker.bindPopup(popupContent, {
      maxWidth: 280,
      className: 'dark-popup'
    });
    
    markersLayer.addLayer(marker);
  });
}

function filterMapByRisk(level) {
  addFireRiskMarkers(level);
}
