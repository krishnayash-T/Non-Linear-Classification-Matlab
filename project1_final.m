clear all; close all;
r=10;
w=6;
N=1000;
d=-8;
rng(20);

%Setting the outer radius of the upper moon
r = r+w/2; 
%Randomly initiating points in the polar coordinate system by by selecting
%radial distance to be between 7 and 13 units in a semicircle
Radial=(r-w)*ones(N,1) + rand(N,1)*w;
angle_theta=rand(N,1)*pi;
%Converting polar coordinates
X1=[Radial.*cos(angle_theta) Radial.*sin(angle_theta)];
Y1=ones(N,1);
%Generating bottom moon with a 180 degree phase shift so that it comes as
%an inverted semicircle
Radial=(r-w)*ones(N,1) + rand(N,1)*w;
angle_theta=pi+rand(N,1)*pi;
%Shifting the bottom moon by defining offsets
mov_x=r-(w/2); %move by 10 units
mov_y=-d; %distance of seperation between moon centres in negative y axis
x=[Radial.*cos(angle_theta)+mov_x Radial.*sin(angle_theta)+mov_y];
y=zeros(N,1);

X=[X1;x];
Y=[Y1;y];
r=r-w/2;

%plotting the clusters
figure(1);
plot(X(Y==1,1),X(Y==1,2),'r+'); hold on;
plot(X(Y==0,1),X(Y==0,2),'bx'); 
axis tight;
axis equal;
title(['Double Moon with: r=' num2str(r) ' w=' num2str(w) ' d=' num2str(d) ' N=' num2str(N)]);

%Creating Training matrix
training_matrix=[X Y];
training_matrix=training_matrix';
rng(10);
training_matrix = training_matrix(:,randperm(length(training_matrix)));

final_train_matrix=training_matrix(1:2,:);
target_matrix=training_matrix(3,:);

%Creating testing data
rng(30);

N1=500;
r = r+w/2; 
Radial1=(r-w)*ones(N1,1) + rand(N1,1)*w;
angle_theta1=rand(N1,1)*pi;
X2=[Radial1.*cos(angle_theta1) Radial1.*sin(angle_theta1)];
Y2=ones(N1,1);
Radial1=(r-w)*ones(N1,1) + rand(N1,1)*w;
angle_theta1=pi+rand(N1,1)*pi;
mov_x=r-(w/2);
mov_y=-d;
x1=[Radial1.*cos(angle_theta1)+mov_x Radial1.*sin(angle_theta1)+mov_y];
y1=zeros(N1,1);
X3=[X2;x1];
Y3=[Y2;y1];
r=r-w/2;

hidn=100;
learnrate=0.1;
rng(6);
net1=feedforwardnet(hidn,'trainlm');
rng(6);
net2=feedforwardnet(hidn,'traingd');
rng(6);
net3=feedforwardnet(hidn,'traingdm');

net1 = configure(net1,final_train_matrix,target_matrix);
net2 = configure(net2,final_train_matrix,target_matrix);
net3 = configure(net3,final_train_matrix,target_matrix);

rng(15);
b=(sqrt(6))/sqrt(hidn+2);
net1.iw{1,1} = (-b + (2*b)*rand(hidn,2));
net2.iw{1,1} = (-b + (2*b)*rand(hidn,2));
net3.iw{1,1} = (-b + (2*b)*rand(hidn,2));


net1.trainParam.lr=learnrate; 
%Set maximum epochs to 5000 
net1.trainParam.epochs=5000; 

net1.trainParam.min_grad =1e-5;
net1.divideParam.trainRatio = 70/100;
net1.divideParam.valRatio = 15/100;
net1.divideParam.testRatio = 15/100;
[net1,tr1] = train(net1,final_train_matrix, target_matrix);
y1=net1(final_train_matrix);
%figure(2);
%plotperform(tr1);
%title(['Levenberg–Marquardt having HiddenNeurons =' num2str(hidn) 'and Learning Rate = ' num2str(learnrate)]);

%Setting the learning rate 
net2.trainParam.lr=learnrate; 

%Set maximum epochs to 5000 
net2.trainParam.epochs=5000; 
net2.trainParam.min_grad =1e-5;
net2.divideParam.trainRatio = 70/100;
net2.divideParam.valRatio = 15/100;
net2.divideParam.testRatio = 15/100;
[net2 , tr2] = train(net2, final_train_matrix, target_matrix);
y2=net2(final_train_matrix);
%figure(3);
%plotperform(tr2);
%title(['Back-Propagation having HiddenNeurons =' num2str(hidn) 'and Learning Rate = ' num2str(learnrate)]);


%Setting the learning rate  
net3.trainParam.lr=learnrate; 

%Set maximum epochs to 5000 
net3.trainParam.epochs=5000; 
net3.trainParam.min_grad =1e-5;
net3.divideParam.trainRatio = 70/100;
net3.divideParam.valRatio = 15/100;
net3.divideParam.testRatio = 15/100;
[net3,tr3] = train(net3, final_train_matrix, target_matrix);
y3=net3(final_train_matrix);
figure(2);
plotperform(tr1);
hold on;
plotperform(tr2);

title(['Back-propagation with momentum having HiddenNeurons =' num2str(hidn) 'and Learning Rate = ' num2str(learnrate)]);

out1=net1(X3');
out2=net2(X3');
out3=net3(X3');

figure(3);
plotconfusion(Y3',out1);
title('Testing Data for Levenberg–Marquardt');

figure(4);
plotconfusion(Y3',out2);
title('Testing Data for Back-propagation ');

figure(5);
plotconfusion(Y3',out3);
title('Testing Data for Back-propagation with momentum');

%Plotting non linear plane
figure(6);
subplot(2,1,1);
cla;
hold on;
title('Decision region for Levenberg–Marquardt');
margin = 0.05; step = 0.5;
td = X3';
xlim([min(td(1,:))-margin max(td(1,:))+margin]);
ylim([min(td(2,:))-margin max(td(2,:))+margin]);
bound =0.5;
hold on;
for x = min(td(1,:))-margin : step : max(td(1,:))+margin
   for y = min(td(2,:))-margin : step : max(td(2,:))+margin
   in_td1 = [x y]';
   net_out = net1(in_td1);
    if(net_out(1)>=bound)
        plot(x, y, 'y.', 'markersize', 18);  
    elseif (net_out(1)<bound)
        plot(x, y, 'g.', 'markersize', 18);
    end
  end
end
plot(X3(Y3==1,1),X3(Y3==1,2),'r.'); hold on;
plot(X3(Y3==0,1),X3(Y3==0,2),'b.'); 
