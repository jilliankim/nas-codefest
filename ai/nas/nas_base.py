import os
import adanet
import tensorflow as tf

# ensembler
def build_AutoEnsembleEstimator(head_type, candidate_pool_list, args):
    return adanet.AutoEnsembleEstimator(
            head=head_type,
            candidate_pool=candidate_pool_list,
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=1000,
                model_dir=args['model_dir']
            ),
            max_iteration_steps=5000
        )


def make_config(args):
    # Estimator configuration.
    return tf.estimator.RunConfig(
    save_checkpoints_steps=1000,
    save_summary_steps=1000,
    tf_random_seed=args['random_seed'],
    model_dir=os.path.join(args['log_dir'], args['experiment_name']))
