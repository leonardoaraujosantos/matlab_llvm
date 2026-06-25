% moving_vehicle.m — sim3d command-line demo, a vehicle driving forward.
%
% Mirrors the MathWorks sim3d "moving vehicle" example using our primitive
% shapes: a box "vehicle" advances along +X at constant speed while a static
% ground-plane actor sits beneath it. Exports a Babylon.js HTML player.
%
%     matlabc -repl < moving_vehicle.m
%     # then open moving_vehicle.html in a browser

w = sim3d.World();

ground = sim3d.Actor('ground', 'plane');
ground.Size = [40 40 1];
ground.Color = [0.18 0.19 0.22];
w.add(ground);

car = sim3d.Actor('vehicle', 'box');
car.Size = [2.0 1.0 0.6];
car.Color = [0.85 0.2 0.2];
w.add(car);

w.open();
sampleTime = 0.02;     % 50 FPS
speed = 15;            % metres / second
x = 0;
for k = 1:120
    x = x + speed * sampleTime;
    car.Translation = [x 0 0.3];
    w.run(sampleTime);
end
w.close();

sim3d.export(w, 'moving_vehicle.html');
disp('wrote moving_vehicle.html');
