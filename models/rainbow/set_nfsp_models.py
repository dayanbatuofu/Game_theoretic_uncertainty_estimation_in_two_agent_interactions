from rainbow.arguments import get_args
from rainbow.model import NFSP_Model, NFSP_Policy

action_size = 5


def get_models():
    args = get_args()

    tag = 'models/rainbow/trained_models/'
    model_index = 5000000
    na_na_path = tag + 'na_na_1593622586'
    a_na_path = tag + 'a_na_1593622607'
    a_a_path = tag + 'a_a_1593622476'

    # na_na_path = tag + '0.6_loss_new/na_na'
    # a_na_path = tag + '0.6_loss_new/a_na' #using alternative loss function for smaller negative values
    # a_a_path = tag + '0.6_loss_new/a_a'

    Q_na_na = NFSP_Model(action_size).to(args.device)
    Q_na_na.load_nfsp_Q(checkpoint_path=na_na_path + '/player_1_' + str(model_index) + '.pth', num=1)

    policy_na_na = NFSP_Policy(action_size).to(args.device)
    policy_na_na.load_nfsp_policy(checkpoint_path=na_na_path + '/player_1_' + str(model_index) + '.pth', num=1)

    Q_na_na_2 = NFSP_Model(action_size).to(args.device)
    Q_na_na_2.load_nfsp_Q(checkpoint_path=na_na_path + '/player_2_' + str(model_index) + '.pth', num=2)

    policy_na_na_2 = NFSP_Policy(action_size).to(args.device)
    policy_na_na_2.load_nfsp_policy(checkpoint_path=na_na_path + '/player_2_' + str(model_index) + '.pth', num=2)

    Q_na_a = NFSP_Model(action_size).to(args.device)
    Q_na_a.load_nfsp_Q(checkpoint_path=a_na_path + '/player_2_' + str(model_index) + '.pth', num=2)

    policy_na_a = NFSP_Policy(action_size).to(args.device)
    policy_na_a.load_nfsp_policy(checkpoint_path=a_na_path + '/player_2_' + str(model_index) + '.pth', num=2)

    Q_a_na = NFSP_Model(action_size).to(args.device)
    Q_a_na.load_nfsp_Q(checkpoint_path=a_na_path + '/player_1_' + str(model_index) + '.pth', num=1)

    policy_a_na = NFSP_Policy(action_size).to(args.device)
    policy_a_na.load_nfsp_policy(checkpoint_path=a_na_path + '/player_1_' + str(model_index) + '.pth', num=1)

    Q_a_a = NFSP_Model(action_size).to(args.device)
    Q_a_a.load_nfsp_Q(checkpoint_path=a_a_path + '/player_1_' + str(model_index) + '.pth', num=1)

    policy_a_a = NFSP_Policy(action_size).to(args.device)
    policy_a_a.load_nfsp_policy(checkpoint_path=a_a_path + '/player_1_' + str(model_index) + '.pth', num=1)

    Q_a_a_2 = NFSP_Model(action_size).to(args.device)
    Q_a_a_2.load_nfsp_Q(checkpoint_path=a_a_path + '/player_2_' + str(model_index) + '.pth', num=2)

    policy_a_a_2 = NFSP_Policy(action_size).to(args.device)
    policy_a_a_2.load_nfsp_policy(checkpoint_path=a_a_path + '/player_2_' + str(model_index) + '.pth', num=2)

    Q_na_na.eval(), Q_na_na_2.eval(), Q_na_a.eval(), Q_a_na.eval(), Q_a_a.eval(), Q_a_a_2.eval(),
    policy_na_na.eval(), policy_na_na_2.eval(), policy_na_a.eval(), policy_a_na.eval(),
    policy_a_a.eval(), policy_a_a_2.eval()

    return (Q_na_na, Q_na_na_2, Q_na_a, Q_a_na, Q_a_a, Q_a_a_2), \
           (policy_na_na, policy_na_na_2, policy_na_a, policy_a_na, policy_a_a, policy_a_a_2)
