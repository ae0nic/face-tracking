from collections.abc import MutableSequence
from typing import Callable


def increment_binary(x : str):
    if not '0' in x: return x
    return ('{:0' + str(len(x)) + 'b}').format(1 + int(x, 2))

def decrement_binary(x : str):
    if not '1' in x: return x
    return ('{:0' + str(len(x)) + 'b}').format(int(x, 2) - 1)

def optimize(parameters : MutableSequence[str],
             decoder : Callable[[MutableSequence[str]], MutableSequence[float]],
             cost : Callable[[MutableSequence[float]], float]) -> MutableSequence[str]:
    decoded_matrix = decoder(parameters)
    result_matrix = []
    for i in range(len(parameters)):

        row = parameters[i]
        min_cost = None
        min_bit = None
        min_binary = None
        for j in [0, 1]:
            temp_decoded = decoded_matrix
            candidate_row = row + str(j)
            temp_decoded[i] = decoder([candidate_row])[0]
            candidate_cost = cost(temp_decoded)
            if min_cost is None or candidate_cost < min_cost:
                min_cost = candidate_cost
                min_bit = j
                decoded_matrix = temp_decoded
                min_binary = candidate_row

        while True:
            temp_decoded = decoded_matrix
            altered_string = min_binary

            # Increment
            if min_bit == 0:
                altered_string = decrement_binary(altered_string)
            else: # Decrement
                altered_string = increment_binary(altered_string)

            temp_decoded[i] = decoder([altered_string])[0]
            candidate_cost = cost(temp_decoded)
            if candidate_cost < min_cost:
                min_binary = altered_string
                min_cost = candidate_cost
            else:
                break
        result_matrix.append(min_binary)
    return result_matrix

def decoder_machine(scale : float, bias : float) -> Callable[[MutableSequence[str]], MutableSequence[float]]:
    def decoder(x : MutableSequence[str]) -> MutableSequence[float]:
        result = []
        for p in x:
            b_sum = 0
            for b in range(len(p) - 1, -1, -1):
                pos = len(p) - b - 1
                b_sum += int(p[b]) * 2**pos
            result.append((scale * b_sum / (2**(len(p)) - 1)) + bias)
        return result
    return decoder

if __name__ == "__main__":
    cost_1 : Callable[[MutableSequence[float]], float] = lambda x : (4*x[0]**2 - 2.1*x[0]**4 +
                                                                     (1/3)*x[0]**6 + x[0]*x[1] -
                                                                     4 * x[1]**2 + 4*x[1]**4)
    cost_2 : Callable[[MutableSequence[float]], float] = lambda x: x[0] * x[1]
    decoder_1 = decoder_machine(180, -90)


    output = optimize(["0", "0"], decoder_1, cost_1)
    for i in range(15):
        output = optimize(output, decoder_1, cost_1)
    print(decoder_1(output))
    print(cost_1(decoder_1(output)))