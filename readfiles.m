clear all
close all
clc

a = textread('filtered.txt'); 


negIndx = find(a(:,1)==-1).';
posIndx = find(a(:,1)==1).';

figure
for i = 1:length(negIndx)

    plot(a(negIndx(i),2), a(negIndx(i),3), 'ro')
    hold on
end

for i = 1:length(posIndx)

    plot(a(posIndx(i),2), a(posIndx(i),3), 'bx')
    hold on
end
title('the libsvm test points set')
grid on 



