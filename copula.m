% Load the stock daily prices 
adj_close = readtable('copula_prices.csv', 'ReadRowNames', true);

% Identify the tickers and the dates 
tickers = adj_close.Properties.VariableNames';

% Calculate the stocks' weekly EXCESS returns
prices  = table2array(adj_close);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);

% apply ksdensity to get a density
for i = 1:size(returns, 2)
    d(:, i) = ksdensity(returns(:, i), returns(:, i), 'function', 'cdf');
end

% fit the copula parameters
[Rho,nu] = copulafit('t', d, 'Method', 'ApproximateML')


% generate a random sample
r = copularnd('t', Rho, nu, 10000);


% apply ksdensity to recover the original scale of returns
for i = 1:size(returns, 2)
    s(:, i) = ksdensity(returns(:, i), r(:, i), 'function', 'icdf');
end

% export to csv
csvwrite('simulate_copulas.csv',s)