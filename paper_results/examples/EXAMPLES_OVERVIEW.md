# Paper Result Examples

## Low / High Semantic Description Examples

| image | label | low semantic | high semantic | structured caption |
|---|---|---|---|---|
| n02951358_31190 | canoe or boat | Visual attributes of the canoe or boat: pink. | Scene and context for the canoe or boat: background, river. | The category is canoe or boat; it shows a woman in a pink kayak navigates a river with rocks in the background. |
| n02951358_34807 | canoe or boat | Visual attributes of the canoe or boat: blue, white. | Scene and context for the canoe or boat: the kayaker is wearing a blue life jacket and a white helmet. | The category is canoe or boat; it shows the kayaker is wearing a blue life jacket and a white helmet. |
| n03452741_5499 | piano | Visual attributes of the piano: red. | Scene and context for the piano: a man in a red shirt sits at a piano in front of a Christmas tree. | The category is piano; it shows a man in a red shirt sits at a piano in front of a Christmas tree. |
| n03376595_247 | chair | Visual attributes of the chair: the chairs are arranged in rows and come in a variety of colors and styles. | Scene and context for the chair: the chairs are arranged in rows and come in a variety of colors and styles. | The category is chair; it shows the chairs are arranged in rows and come in a variety of colors and styles. |
| n02906734_14010 | brush | Visual attributes of the brush: a man in a hat is skating on the ice with a broom. | Scene and context for the brush: a man in a hat is skating on the ice with a broom. | The category is brush; it shows a man in a hat is skating on the ice with a broom. |

## Classifier-guided Caption Output Examples

| image | true | classifier/main | generated caption |
|---|---|---|---|
| n03452741_16744 | piano | handbag | The category is handbag; it shows a black bag with a pink and purple geometric design. |
| n04069434_10318 | camera | camera | The category is camera; it shows a black, wooden device in a collage of various Sony digital cameras. |
| n03452741_2982 | piano | piano | The category is piano; it shows a grand piano with black and white keys. |
| n03180011_9576 | computer | panda or pandas | The category is panda or pandas; it shows a panda bear sitting on a tree branch, eating bamboo. |
| n03773504_6291 | rocket | bicycle | The category is bicycle; it shows a red bicycle parked on a sidewalk near a crosswalk, next to a tall building. |

These examples illustrate the current inference design: the EEG classifier selects the main object category, while low/high retrieved semantic evidence supplies visual attributes and scene context.
