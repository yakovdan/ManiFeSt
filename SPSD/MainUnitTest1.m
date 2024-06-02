close all
clear

%%
d  = 10;
r  = 4;
N1 = 20;
N2 = 25;

%% Generate data:
mCC1{N1} = [];
for ii = 1 : N1
    mM       = randn(d, r);
    mCC1{ii} = mM * mM';
end

mCC2{N2} = [];
for ii = 1 : N2
    mM       = randn(d, r);
    mCC2{ii} = mM * mM';
end

%% Apply DA:
mCC1Tilde = ApplySpsdPT(mCC1, mCC2, r);

%%
% mMean1      = SpsdMean(mCC1, r);
mMean2      = SpsdMean(mCC2, r);
mMean1Tilde = SpsdMean(mCC1Tilde, r);

%%
norm(mMean2 - mMean1Tilde, 'inf')


