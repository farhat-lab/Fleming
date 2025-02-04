# from paperqa import Settings, ask
import pqapi
import argparse


def run_paperqa(molecule_description,
                paper_directory,
                save_dir):
    # answer = ask(
    #     molecule_description, 
    #     settings=Settings(temperature=0.5, paper_directory=paper_directory),
    # )

    answer = pqapi.agent_query(molecule_description)

    answer = str(answer.answer)

    save_file = open(save_dir, 'w')
    save_file.write(answer)
    save_file.close

    return answer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--molecule_description', type=str,
                        help = "description of the molecule")
    
    parser.add_argument('--paper_directory', type=str,
                        help = "Directory for papers")
    
    parser.add_argument('--save_dir', type=str,
                        help = "Directory for saving results")
    
    args = parser.parse_args()

    run_paperqa(molecule_description=args.molecule_description,
                paper_directory=args.paper_directory,
                save_dir=args.save_dir)