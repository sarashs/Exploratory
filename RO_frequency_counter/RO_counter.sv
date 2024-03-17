`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/16/2024 11:26:29 PM
// Design Name: 
// Module Name: RO_counter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module RO_counter #(
    parameter int num_counters = 1
)
(
    input logic reset, // Added reset signal, active high for simplicity.
    input logic [num_counters-1:0] in_signal,
    input logic clk,
    input logic pause,
    output logic [num_counters*32 - 1:0] freq
);

logic [num_counters*32 - 1 :0] freq_count;
logic [31:0] clk_count;
logic clk_done;
logic [num_counters*32 - 1:0] freq_out;

assign freq = freq_out;

// Incorporate reset logic in procedural blocks.
always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
        clk_count <= 0;
        clk_done <= 1;
    end
    else if (clk_count >= 10) begin
        clk_done <= 1;
        clk_count <= 0;
    end 
    else begin
        clk_count <= clk_count + 1;
        clk_done <= 0;
    end
end

// Add reset logic for the frequency counters.
for (genvar i = 0; i < num_counters; i++) begin : gen_freq_counters
    always_ff @(posedge in_signal[i] or posedge reset) begin
        if (reset || clk_done) begin
            freq_count[(i+1)*32 - 1:i*32] <= 0;
        end
        else if (!clk_done) begin
            freq_count[(i+1)*32 - 1:i*32] <= freq_count[(i+1)*32 - 1:i*32] + 1;
        end
    end
end

// Handle pause and reset for frequency output update.
always_ff @(posedge clk_done or posedge reset) begin
    if (reset) begin
        freq_out <= 0;
    end
    else if (!pause) begin
        freq_out <= freq_count;
    end
end

endmodule