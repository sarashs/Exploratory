`timescale 1ns/1ps

module RO_counter_tb;

parameter int num_counters = 1;
logic reset, clk, pause;
logic [num_counters-1:0] in_signal;
logic [num_counters*32 - 1:0] freq;

// Instantiate the DUT (Device Under Test)
RO_counter #(
    .num_counters(num_counters)
) dut (
    .reset(reset),
    .in_signal(in_signal),
    .clk(clk),
    .pause(pause),
    .freq(freq)
);

// Generate 100 MHz clock (10 ns period)
initial begin
    clk = 0;
    forever #5 clk = ~clk; // Toggle every 5 ns for a 10 ns period
end

// Correctly generate in_signal with frequency change
initial begin
    in_signal = 0;
    // Initial period for 400 MHz, toggle every 1.25 ns for a 2.5 ns period
    // Wait for 250 ns to change frequency to 320 MHz, toggle every 1.5625 ns for a 3.125 ns period
    repeat (200) begin // 200 * 1.25 ns = 250 ns, before changing frequency
        #1.25 in_signal = ~in_signal;
    end
    forever #1.5625 in_signal = ~in_signal; // Now toggle every 1.5625 ns for a 3.125 ns period
end

// Set pause after 50 clk cycles (500 ns)
initial begin
    pause = 0;
    #500 pause = 1; // Set pause after 500 ns
end

// Initialize testbench variables and run the simulation for a certain time
initial begin
    reset = 1;
    #20 reset = 0; // Release reset after 20 ns
    #1000 $finish; // Run simulation for 1000 ns then finish
end

// Optional: Monitor signals
initial begin
    $monitor("Time=%t clk=%b in_signal=%b pause=%b freq=%h", $time, clk, in_signal, pause, freq);
end

endmodule
