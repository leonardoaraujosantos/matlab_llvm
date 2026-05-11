% geo_helpers.m
% ====================================================================
% Geographic helpers: Haversine + Vincenty distances, initial bearing,
% great-circle destination. PROP-Tier-1a §3.1.5.
% ====================================================================

% Famous fixed pairs
% London Heathrow -> JFK New York
lat1 = 51.4700; lon1 = -0.4543;
lat2 = 40.6413; lon2 = -73.7781;
dh   = haversine(lat1, lon1, lat2, lon2);
dv   = vincenty  (lat1, lon1, lat2, lon2);
az   = bearing   (lat1, lon1, lat2, lon2);
fprintf('LHR -> JFK\n');
fprintf('  Haversine : %.1f km\n', dh / 1000.0);
fprintf('  Vincenty  : %.1f km\n', dv / 1000.0);
fprintf('  Bearing   : %.2f deg compass\n', az);

% Sao Paulo -> Tokyo
lat1 = -23.5505; lon1 = -46.6333;
lat2 =  35.6762; lon2 = 139.6503;
dh = haversine(lat1, lon1, lat2, lon2);
dv = vincenty (lat1, lon1, lat2, lon2);
az = bearing  (lat1, lon1, lat2, lon2);
fprintf('\nSao Paulo -> Tokyo\n');
fprintf('  Haversine : %.1f km\n', dh / 1000.0);
fprintf('  Vincenty  : %.1f km\n', dv / 1000.0);
fprintf('  Bearing   : %.2f deg compass\n', az);

% Destination point: 100 km east of Bridgetown, Barbados
src_lat = 13.0975; src_lon = -59.6133;
d_m     = 100e3;
az_east = 90.0;
dst_lat = greatCircleDestLat(src_lat, src_lon, d_m, az_east);
dst_lon = greatCircleDestLon(src_lat, src_lon, d_m, az_east);
fprintf('\n100 km east of Bridgetown (13.10 N, -59.61 W)\n');
fprintf('  Destination : %.4f N, %.4f E\n', dst_lat, dst_lon);
% Validate by haversine round-trip
d_check = haversine(src_lat, src_lon, dst_lat, dst_lon);
fprintf('  Round-trip distance check: %.2f km\n', d_check / 1000.0);
