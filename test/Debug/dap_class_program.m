% Fixture for class-instance Locals + Watch. The breakpoint at the
% last line gives the script frame three class-bound script vars
% (acc, sav, p) of two distinct classes (Account, Savings) so the
% scenario can exercise both registry hit-paths and property
% expansion. Line numbers are referenced by dap_scenarios.py.
acc = Account(101, 50);
acc.deposit(25);
sav = Savings(202, 100, 0.10);
p = Account(303, 0);
disp(p.Id);

classdef Account
    properties
        Id
        Balance
    end
    methods
        function obj = Account(id, bal)
            if nargin == 2
                obj.Id = id;
                obj.Balance = bal;
            end
        end
        function deposit(obj, amt)
            obj.Balance = obj.Balance + amt;
        end
    end
end

classdef Savings < Account
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
    end
end
