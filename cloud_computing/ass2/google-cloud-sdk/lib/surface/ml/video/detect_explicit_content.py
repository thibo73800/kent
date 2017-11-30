# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command to analyze explicit content in videos."""

from googlecloudsdk.api_lib.ml.video import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml.video import video_command_util


@base.ReleaseTracks(base.ReleaseTrack.BETA)
class DetectExplicitContent(base.Command):
  """Detect explicit content in videos.

  Detect adult content within a video. Adult content is content generally
  appropriate for 18 years of age and older, including but not limited to
  nudity, sexual activities, and pornography (including cartoons or anime).

  The response includes a bucketized likelihood value, from VERY_UNLIKELY to
  VERY_LIKELY. When Explicit Content Detection evaluates a video, it does so on
  a per-frame basis and considers visual content only (not audio).

  {auth_help}
  """
  detailed_help = {'auth_help': video_command_util.SERVICE_ACCOUNT_HELP}

  @staticmethod
  def Args(parser):
    parser.display_info.AddFormat('json')
    video_command_util.AddVideoFlags(parser)

  def Run(self, args):
    """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. Includes all the arguments that were provided
        to this command invocation.

    Raises:
      video_client.VideoUriFormatError: if the input path is invalid.
      video_client.SegmentError: if the segments can't be parsed.

    Returns:
      videointelligence_v1beta1_messages.GoogleLongRunningOperation |
      videointelligence_v1beta1_messages.
      GoogleCloudVideointelligenceV1AnnotateVideoResponse: the name of the
        operation if --async is given, otherwise the result of the analysis.
    """
    return util.AnnotateVideo('EXPLICIT_CONTENT_DETECTION',
                              args.input_path,
                              output_uri=args.output_uri,
                              segments=args.segments,
                              region=args.region,
                              async=args.async)
