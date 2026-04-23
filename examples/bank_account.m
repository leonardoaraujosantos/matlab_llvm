% A small classdef example: a bank account with a handful of
% behaviours you can touch through the compiler's current OOP MVP.
%
% Shows off:
%   - `classdef Name` with `properties` and `methods`
%   - constructor that reads `nargin` to make the id/deposit optional
%   - instance methods called via dot-syntax (acc.deposit(100))
%   - a `Dependent` property (`Overdrawn`) with a `get.Prop` method
%     that computes on read
%   - operator overloading (`a == b` dispatches to AccountId__eq)
%   - a Savings subclass that adds an `interest` method and inherits
%     everything else from BankAccount

acc = BankAccount(1001, 500);
disp(acc.Id);                 % 1001
disp(acc.Balance);            % 500
disp(acc.Overdrawn);          % 0 — dependent property

acc.deposit(250);
disp(acc.Balance);            % 750

acc.withdraw(1000);
disp(acc.Balance);            % -250
disp(acc.Overdrawn);          % 1

other = BankAccount(1001, 0);
disp(acc == other);           % 1 — same id via operator==

sav = Savings(2000, 1000, 0.05);
disp(sav.interest());         % 50  — 5% of 1000

classdef BankAccount
    properties
        Id
        Balance
    end
    properties (Dependent)
        Overdrawn
    end
    methods
        function obj = BankAccount(id, bal)
            if nargin == 2
                obj.Id = id;
                obj.Balance = bal;
            end
        end
        function deposit(obj, amt)
            obj.Balance = obj.Balance + amt;
        end
        function withdraw(obj, amt)
            obj.Balance = obj.Balance - amt;
        end
        function f = get.Overdrawn(obj)
            if obj.Balance < 0
                f = 1;
            else
                f = 0;
            end
        end
        function r = eq(a, b)
            if a.Id == b.Id
                r = 1;
            else
                r = 0;
            end
        end
    end
end

classdef Savings < BankAccount
    properties
        Rate
    end
    methods
        function obj = Savings(id, bal, rate)
            if nargin == 3
                obj.Id = id;
                obj.Balance = bal;
                obj.Rate = rate;
            end
        end
        function i = interest(obj)
            i = obj.Balance * obj.Rate;
        end
    end
end
