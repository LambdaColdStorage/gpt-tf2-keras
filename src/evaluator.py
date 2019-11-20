import numpy as np


from src import utils


def eval_cnndm(args, ds, model, enc):
    cur_v = 0
    max_v = 100000000

    with open(args.output_file, 'w') as f:
        for (value, num_samples) in ds:
            print('Sample {0} out of {1}'.format(cur_v, num_samples))
            input_data = [np.array(value).tolist() for i in range(args.num_trials)]
            start_length = [len(value) for data in input_data]
            flag_stop = [False for i in range(args.num_trials)]
            idx_stop = [args.length for i in range(args.num_trials)]

            for shift in range(args.output_length):
                output_data = model.predict(np.array(input_data))
                
                for index in range(args.num_trials):

                    if not flag_stop[index]:
                        probs = [(prob, i) for i, prob in enumerate(
                            output_data[index, start_length[index] + shift - 1])]
                        probs.sort(reverse=True)
                        
                        if args.nucleus:
                            next_token = utils.find_top_p(probs, args.top_p, args.temperature)
                        else:
                            next_token = utils.find_top_k(probs, args.top_k, args.temperature)

                        input_data[index].append(next_token)

                        if next_token == 50256:
                            flag_stop[index] = True
                            if idx_stop[index] == args.length:
                                idx_stop[index] = len(input_data[index]) - 1
                    else:
                        input_data[index].append(50256)

            # print result
            line = ''
            for index in range(args.num_trials):
                output = enc.decode(input_data[index])
                output_tldr = enc.decode(input_data[index][start_length[index]:idx_stop[index]])
                output_tldr = output_tldr.replace('\n', ' ')
                line += '<t> ' + output_tldr + ' </t>'
                if index == args.num_trials - 1:
                    line += '\n'
                else:
                    line += ' '
                print(output_tldr)
                print('------------------------------------------------')
            f.write(line)
            f.flush()

            cur_v += 1
            if cur_v == max_v:
                break
            print('==================================================================') 


def eval(args, ds, model, enc):

    if args.task == "cnndm":
        eval_cnndm(args, ds, model, enc)
    elif args.task == "coqa":
        eval_coqa(args, ds, model, enc)
    else:
        print("Evaluation task " + args.task + " has not been implemented")
