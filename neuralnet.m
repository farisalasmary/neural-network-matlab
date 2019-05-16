
%
%    @author
%          ______         _                  _
%         |  ____|       (_)           /\   | |
%         | |__ __ _ _ __ _ ___       /  \  | | __ _ ___ _ __ ___   __ _ _ __ _   _
%         |  __/ _` | '__| / __|     / /\ \ | |/ _` / __| '_ ` _ \ / _` | '__| | | |
%         | | | (_| | |  | \__ \    / ____ \| | (_| \__ \ | | | | | (_| | |  | |_| |
%         |_|  \__,_|_|  |_|___/   /_/    \_\_|\__,_|___/_| |_| |_|\__,_|_|   \__, |
%                                                                              __/ |
%                                                                             |___/
%            Email: farisalasmary@gmail.com
%            Date:  May 11, 2019
%

addpath(pwd); % to make Octave happy

function g = sigmoid(z)
    g = 1 ./ (1 + e.^-z);
end

function g = sigmoid_prime(z)
    g = z .* (1 - z);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.9
epochs = 50000


% load data
train_data = dlmread ('AHDBase_TrainingSet.csv', ',', 1, 1);
test_data = dlmread ('AHDBase_TestingSet.csv', ',', 1, 1);

% get X_train, y_train, X_test, and y_test
X_train = train_data(:, 1:64);
y_train = train_data(:, 65);

X_test = test_data(:, 1:64);
y_test = test_data(:, 65);

m = size(X_train)(1);

% weight initialization
W1 = 2 * rand(32, 64) - 1;
W1 /= 32;

b1 = 2 * rand(32, 1) - 1;

W2 = 2 * rand(16, 32) - 1;
W2 /= 16;

b2 = 2 * rand(16, 1) - 1;

W3 = 2 * rand(10, 16) - 1;
W3 /= 10;

b3 = 2 * rand(10, 1) - 1;

% one-hot encoding
y_true = full(sparse(1:numel(y_train), y_train+1, 1, numel(y_train), 10));

% take transpose for our matrices since the network deals with columns as samples and rows as features
A0 = X_train';
y_true = y_true';

costs = 0;
for i = 1:epochs
    % feedforward
    Z1 = W1 * A0 + b1;
    A1 = sigmoid(Z1);

    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);

    Z3 = W3 * A2 + b3;
    A3 = sigmoid(Z3);

    y_pred = A3;

    cost = 0.5 * sum(sum((y_true - y_pred) .^ 2)) / m;

    display(i);
    display(cost);
    costs(i) = cost;

    % backpropagate error
    delta3 = -(y_true - y_pred) .* sigmoid_prime(y_pred);
    delta2 = (W3' * delta3) .* sigmoid_prime(A2);
    delta1 = (W2' * delta2) .* sigmoid_prime(A1);

    dW1 = (delta1 * A0') / m;
    dW2 = (delta2 * A1') / m;
    dW3 = (delta3 * A2') / m;

    db1 = sum(delta1, 2) / m;
    db2 = sum(delta2, 2) / m;
    db3 = sum(delta3, 2) / m;


    % update the weights
    W1 = W1 - alpha*dW1;
    W2 = W2 - alpha*dW2;
    W3 = W3 - alpha*dW3;

    b1 = b1 - alpha*db1;
    b2 = b2 - alpha*db2;
    b3 = b3 - alpha*db3;
endfor


%%%%%%%% Testing %%%%%%%%
A0 = X_test';
% feedforward
Z1 = W1 * A0 + b1;
A1 = sigmoid(Z1);

Z2 = W2 * A1 + b2;
A2 = sigmoid(Z2);

Z3 = W3 * A2 + b3;
A3 = sigmoid(Z3);

y_pred = A3;

# convert one-hot encoding back into labels by taking argmax for predicted outputs
[max_values labels] = max(y_pred);

# if the output are the same that means our model is doing the job
display(labels(1));
display(y_test(1));
