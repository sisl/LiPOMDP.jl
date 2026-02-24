using Test
using LitPOMDP
using POMDPs

@testset "LitPOMDP" begin

    @testset "Package loads" begin
        @test true  # if we got here, the package loaded
    end

    @testset "Struct construction" begin
        s = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 1.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test s.time == 1.0
        @test length(s.deposit) == 4

        a = Action(site = [0, 0, 0, 0], type = [0, 0, 0, 0])
        @test a.site == [0, 0, 0, 0]
        @test a.type == [0, 0, 0, 0]

        o = Observation(deposits = [1.0, 2.0, 3.0, 4.0],
                        observed_price = 10_000.0,
                        observed_demand = 5_000.0)
        @test o.observed_price == 10_000.0
    end

    @testset "LiPOMDP model" begin
        pomdp = LiPOMDP()
        @test pomdp.n_deposits == 4
        @test pomdp.time_horizon == 29
        @test 0.0 < POMDPs.discount(pomdp) < 1.0

        # Custom parameters
        pomdp2 = LiPOMDP(price_model_type = 4, alpha = 0.8)
        @test pomdp2.price_model_type == 4
        @test pomdp2.alpha == 0.8
    end

    @testset "Initial belief and state" begin
        pomdp = LiPOMDP()
        s0, belief = create_initial_belief(pomdp)

        @test length(s0.deposit) == 4
        @test s0.time == 1.0
        @test all(.!s0.have_opened)
        @test all(.!s0.is_depleted)
        @test belief.time == 1.0
        @test length(belief.deposits_distribution) == 4
    end

    @testset "Action space" begin
        pomdp = LiPOMDP()
        acts = POMDPs.actions(pomdp)
        @test length(acts) > 0
        @test all(a -> isa(a, Action), acts)
    end

    @testset "Terminal check" begin
        pomdp = LiPOMDP()
        s = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 1.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test !POMDPs.isterminal(pomdp, s)

        # Past time horizon → terminal
        s_late = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 100.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test POMDPs.isterminal(pomdp, s_late)
    end

    @testset "Pricing models" begin
        for model_type in [1, 2, 3, 4]
            pomdp = LiPOMDP(price_model_type = model_type)
            @test pomdp.price_model_type == model_type
            s0, _ = create_initial_belief(pomdp)
            @test s0.price > 0.0
        end
    end

end
