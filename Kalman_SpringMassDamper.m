% Defining the Spring Mass Dampener 

m = 10;
b = 1;
k = 2;

% Defining the continuous-time model

A = [ 0 1; -k/m -b/m];
B = [0; 1/m];
Bw = B;
C = [1 0];
D = 0;

dT = 0.1;  % Creating Discrete Time Model
Ad = expm(A*dT);
Bd = inv(A)*(Ad - eye(2))*B;
Cd = C;
Dd = D;

Sw = 0.01; Sv = 1e-5;    % Specify Noise model
Z = [-A Bw*Sw*Bw'; zeros(2) A'];  %Convert to discrete time
C = expm(Z*dT);
c12 = C(1:2, 3:4);
c22 = C(3:4, 3:4);
SigmaW = c22'*c12;
SigmaV = Sv/dT;

numSims = 1;  % Define the number of simulations to run
fprintf('Finished defining the model\n');

%% Running the Simulation

t = 0:dT:100; % Setting up simulation time vector
u = sin(2*pi*(1/10)*t);  % Set up deterministic system input
nt = length(t); % length of time vector
randn("state",0);

x = zeros(2,nt+1,numSims); % reserving storage for state profiles for all simulations
z = zeros(nt, numSims); % reserving storage for the output for all simulations

for theSim = 1:numSims  % Execute numSims simulations with different random inputs
    for k = 1:length(t) % This is the kth step for each simulation
        x(:,k+1,theSim) = Ad*x(:,k,theSim) + Bd*u(k) + chol(SigmaW,"lower")*randn(2,1);
        z(k,theSim)    = Cd*x(:,k,theSim) + Dd*u(k) + chol(SigmaV,"lower")*randn(1);
    end
end
x = x(:,1:nt,:); % Crop the states to the same length as t
fprintf('Finished running the simulations\n');

%% Plotting

% Plot x1(k), the position state
figure(1)
plot(t,squeeze(x(1,:,:))); grid on
title(sprintf('Position state from %d simulations',numSims));
%fprintf("Final value of position is:");

xlabel('Time (s)'); ylabel('Position (m)');

% Plot x2(k), the velocity state
figure(2)
plot(t,squeeze(x(2,:,:))); grid on
title(sprintf('Velocity state from %d simulations',numSims));
xlabel('Time (s)'); ylabel('Velocity (m/s)');

% Plot z(k), the measured output
figure(3)
plot(t,z); grid on
title(sprintf('Measured output from %d simulations',numSims));
xlabel('Time (s)'); ylabel('Position (m)');

% Final values at last time step
final_position = squeeze(x(1,end,:));   % Position (x1) at final time
final_velocity = squeeze(x(2,end,:));   % Velocity (x2) at final time
final_output   = z(end,:);              % Measured output at final time

% Print the values
fprintf('Final position value(s): %.6f\n', final_position);
fprintf('Final velocity value(s): %.6f\n', final_velocity);
fprintf('Final measured output value(s): %.6f\n', final_output);

%% Running the Kalman Filter

[nx,nt] = size(x);
xhat = zeros(nx,1);        % Initialize state estimate at time zero
SigmaX = zeros(nx,nx);     %  and its covariance matrix
xhatstore = zeros(nx,nt);  % Reserve storage for all state estimates
boundstore = zeros(nx,nt); %  and their confidence bounds

for k = 2:nt               % Execute the Kalman filter
  % KF Step 1a: State estimate time update      
  xhat = Ad*xhat + Bd*u(k-1); % use prior value of "u"
 
  % KF Step 1b: Error covariance time update   
  SigmaX = Ad*SigmaX*Ad' + SigmaW;
  
  % KF Step 1c: Estimate system output   
  zhat = Cd*xhat + Dd*u(k); % use present value of "u"

  % KF Step 2a: Compute Kalman gain matrix   
  L = SigmaX*Cd'/(Cd*SigmaX*Cd' + SigmaV); 
 
  % KF Step 2b: State estimate measurement update   
  xhat = xhat + L*(z(k) - zhat);
   
  % KF Step 2c: Error covariance measurement update   
  SigmaX = (eye(nx)-L*Cd)*SigmaX;
 
  % [Store information for evaluation/plotting purposes]  
  xhatstore(:,k) = xhat; 
  boundstore(:,k) = 3*sqrt(diag(SigmaX));
end

%% Plotting results

% Plot the states with confidence bounds
figure(4)
t2 = [t fliplr(t)]; % Prepare for plotting bounds via "fill"
x2 = [xhatstore-boundstore fliplr(xhatstore+boundstore)];
h1a = fill(t2,x2(1,:),'b',"FaceAlpha",0.05); hold on; grid on
fill(t2,x2(2,:),'b',"FaceAlpha",0.05); 
h2 = plot(t,x',t,xhatstore','--'); ylim([-0.15 0.25]);
legend([h2;h1a],{'True position','True velocity','Position estimate',...
  'Velocity estimate','Bounds'}); 
title('Demonstration of Kalman filter state estimates    '); 
xlabel('Time (s)'); ylabel('State (m or m/s)'); 


figure(5)
xerr = x - xhatstore; 
fill([t fliplr(t)],[-boundstore(1,:) fliplr(boundstore(1,:))],'b',...
  "FaceAlpha",0.05); hold on; grid on;
plot(t,xerr(1,:),'b');
title('Position estimation error with bounds'); 
xlabel('Time (s)'); ylabel('Error (m)'); 

figure(6)
fill([t fliplr(t)],[-boundstore(2,:) fliplr(boundstore(2,:))],'b',...
  "FaceAlpha",0.05); hold on; grid on;
plot(t,xerr(2,:),'b');
title('Velocity estimation error with bounds'); 
xlabel('Time (s)'); ylabel('Error (m/s)'); 